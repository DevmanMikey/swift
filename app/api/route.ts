import { headers } from "next/headers";
import { z } from "zod";
import { zfd } from "zod-form-data";
import { unstable_after as after } from "next/server";
import OpenAI from "openai";

// Custom OpenAI client for router.inspiraus.work
const llmClient = new OpenAI({
    baseURL: "https://router.inspiraus.work/v1",
    apiKey: process.env.INSPIRAUS_API_KEY || "sk-",
});

// DeepInfra client for STT/TTS
const deepInfraClient = new OpenAI({
    baseURL: "https://api.deepinfra.com/v1/openai",
    apiKey: process.env.DEEPINFRA_API_KEY || "sk-",
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

        const completion = await llmClient.chat.completions.create({
                model: "zephyr", // Using the Zephyr model from router.inspiraus.work
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
                        - Your large language model is Zephyr, a powerful open-source model, accessed through router.inspiraus.work.
                        - Your speech-to-text and text-to-speech models are provided by DeepInfra, using their Whisper Turbo and Zephyr models.
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
                "deepinfra tts request " + request.headers.get("x-vercel-id") || "local"
        );

        // Using DeepInfra's API for TTS with their Zephyr model
        const voice = await fetch("https://api.deepinfra.com/v1/inference/microsoft/speecht5_tts", {
                method: "POST",
                headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${process.env.DEEPINFRA_API_KEY}`,
                },
                body: JSON.stringify({
                        inputs: response,
                        parameters: {
                                speaker_embeddings: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin",
                                rate: 1.0,
                                sample_rate: 24000,
                                output_format: "pcm_f32le"
                        }
                }),
        });

        console.timeEnd(
                "deepinfra tts request " + request.headers.get("x-vercel-id") || "local"
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
                const transcription = await deepInfraClient.audio.transcriptions.create({
                        file: input,
                        model: "whisper-large-v3", // DeepInfra's Whisper Turbo model
                        language: "en",
                });

                return transcription.text.trim() || null;
        } catch (error) {
                console.error("STT Error:", error);
                return null; // Empty audio file or error
        }
}
