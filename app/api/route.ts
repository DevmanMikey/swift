import Groq from "groq-sdk";
import { headers } from "next/headers";
import { z } from "zod";
import { zfd } from "zod-form-data";
import { unstable_after as after } from "next/server";
import OpenAI from "openai";

const groq = new Groq();
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const schema = zfd.formData({
        input: z.union([zfd.text(), zfd.file()]),
        message: zfd.repeatableOfType(
                zfd.json(
                        z.object({
                                role: z.enum(["user", "assistant"]),
                                content: z.string(),
                        })
                )
        ),
});

export async function POST(request: Request) {
        console.time("transcribe " + request.headers.get("x-vercel-id") || "local");

        const { data, success } = schema.safeParse(await request.formData());
        if (!success) return new Response("Invalid request", { status: 400 });

        const transcript = await getTranscript(data.input);
        if (!transcript) return new Response("Invalid audio", { status: 400 });

        console.timeEnd(
                "transcribe " + request.headers.get("x-vercel-id") || "local"
        );
        console.time(
                "text completion " + request.headers.get("x-vercel-id") || "local"
        );

        const completion = await openai.chat.completions.create({
                model: "gpt-4o",
                messages: [
                        {
                                role: "system",
                                content: `- You are Swift, a friendly and helpful voice assistant.
                        - Respond briefly to the user's request, and do not provide unnecessary information.
                        - If you don't understand the user's request, ask for clarification.
                        - You do not have access to up-to-date information, so you should not provide real-time data.
                        - You are not capable of performing actions other than responding to the user.
                        - Do not use markdown, emojis, or other formatting in your responses. Respond in a way easily spoken by text-to-speech software.
                        - User location is ${location()}.
                        - The current time is ${time()}.
                        - Your large language model is GPT-4o, created by OpenAI, a powerful multimodal model. 
                        - Your text-to-speech model is ElevenLabs, a company that builds high-quality voice synthesis technology.
                        - You are built with Next.js and hosted on Vercel.`,
                        },
                        ...data.message,
                        {
                                role: "user",
                                content: transcript,
                        },
                ],
        });

        const response = completion.choices[0].message.content;
        console.timeEnd(
                "text completion " + request.headers.get("x-vercel-id") || "local"
        );

        console.time(
                "elevenlabs request " + request.headers.get("x-vercel-id") || "local"
        );

        const voice = await fetch("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM", {
                method: "POST",
                headers: {
                        "Content-Type": "application/json",
                        "xi-api-key": process.env.ELEVENLABS_API_KEY!,
                },
                body: JSON.stringify({
                        text: response,
                        model_id: "eleven_turbo_v2",
                        voice_settings: {
                                stability: 0.5,
                                similarity_boost: 0.75
                        },
                        output_format: "pcm_24000"
                }),
        });

        console.timeEnd(
                "elevenlabs request " + request.headers.get("x-vercel-id") || "local"
        );

        if (!voice.ok) {
                console.error(await voice.text());
                return new Response("Voice synthesis failed", { status: 500 });
        }

        console.time("stream " + request.headers.get("x-vercel-id") || "local");
        after(() => {
                console.timeEnd(
                        "stream " + request.headers.get("x-vercel-id") || "local"
                );
        });

        return new Response(voice.body, {
                headers: {
                        "X-Transcript": encodeURIComponent(transcript),
                        "X-Response": encodeURIComponent(response),
                },
        });
}

function location() {
        const headersList = headers();

        const country = headersList.get("x-vercel-ip-country");
        const region = headersList.get("x-vercel-ip-country-region");
        const city = headersList.get("x-vercel-ip-city");

        if (!country || !region || !city) return "unknown";

        return `${city}, ${region}, ${country}`;
}

function time() {
        return new Date().toLocaleString("en-US", {
                timeZone: headers().get("x-vercel-ip-timezone") || undefined,
        });
}

async function getTranscript(input: string | File) {
        if (typeof input === "string") return input;

        try {
                const transcription = await openai.audio.transcriptions.create({
                        file: input,
                        model: "whisper-1",
                        language: "en",
                });

                return transcription.text.trim() || null;
        } catch {
                return null; // Empty audio file
        }
}
