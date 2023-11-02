import { stdin as input, stdout as output } from "node:process";
// readline/promises is still experimental so not in @types/node yet
// @ts-ignore
import readline from "node:readline/promises";
import fs from "fs/promises";
import dotenv from 'dotenv'; 


import {
  ContextChatEngine,
  Document,
  serviceContextFromDefaults,
  VectorStoreIndex,
} from "llamaindex";


async function main() {
    dotenv.config();

    const essay = await fs.readFile(
        "node_modules/llamaindex/examples/abramov.txt",
        "utf-8",
      );
    
  const document = new Document({ text: essay });
  const serviceContext = serviceContextFromDefaults({ chunkSize: 512 });
  const index = await VectorStoreIndex.fromDocuments([document], {
    serviceContext,
  });
  const retriever = index.asRetriever();
  retriever.similarityTopK = 5;
  const chatEngine = new ContextChatEngine({ retriever });
  const rl = readline.createInterface({ input, output });

  while (true) {
    const query = await rl.question("Query: ");
    const response = await chatEngine.chat(query);
    console.log(response.toString());
  }
}

main().catch(console.error);