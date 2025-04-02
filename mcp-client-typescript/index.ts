import { OpenAI } from "openai";
import { EasyInputMessage, FunctionTool } from "openai/resources/responses/responses";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";

import dotenv from "dotenv";

dotenv.config(); // load environment variables from .env

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is not set");
}

class MCPClient {
  private mcp: Client;
  private transport: StdioClientTransport | null = null;
  private openai: OpenAI;
  private tools: FunctionTool[] = [];

  constructor() {
    // Initialize OpenAI client and MCP client
    this.openai = new OpenAI({
      apiKey: OPENAI_API_KEY,
    });
    this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
  }

  async connectToServer(serverScriptPath: string) {
    /**
     * Connect to an MCP server
     *
     * @param serverScriptPath - Path to the server script (.py or .js)
     */
    try {
      // Determine script type and appropriate command
      const isJs = serverScriptPath.endsWith(".js");
      const isPy = serverScriptPath.endsWith(".py");
      if (!isJs && !isPy) {
        throw new Error("Server script must be a .js or .py file");
      }
      const command = isPy
        ? process.platform === "win32"
          ? "python"
          : "python3"
        : process.execPath;

      // Initialize transport and connect to server
      this.transport = new StdioClientTransport({
        command,
        args: [serverScriptPath],
      });
      this.mcp.connect(this.transport);

      // List available tools
      const toolsResult = await this.mcp.listTools();
      this.tools = toolsResult.tools.map((tool) => {
        return {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
          type: "function",
          strict: false
        };
      });
      console.log(
        "Connected to server with tools:",
        this.tools.map(({ name }) => name),
      );
    } catch (e) {
      console.log("Failed to connect to MCP server: ", e);
      throw e;
    }
  }

  async processQuery(query: string) {
    /**
     * Process a query using the LLM and available tools
     *
     * @param query - The user's input query
     * @returns Processed response as a string
     */
    const input: EasyInputMessage[] = [
      {
        role: "user",
        content: query,
      },
    ];

    const response = await this.openai.responses.create({
      input,
      model: "gpt-4o",
      tools: this.tools
    });

    // Process response and handle tool calls
    const finalText = [];

    for (const output of response.output) {
      if (output.type === "message") {
        finalText.push(response.output_text);
      } else if (output.type === "function_call") {
        // Execute tool call
        const toolName = output.name;
        const toolArgs = JSON.parse(output.arguments) as { [x: string]: unknown } | undefined;

        const result = await this.mcp.callTool({
          name: toolName,
          arguments: toolArgs,
        });
        finalText.push(
          `[Calling tool ${toolName} with args ${JSON.stringify(toolArgs)}]`,
        );

        // Continue conversation with tool results
        input.push({
          role: "user",
          content: (result.content as [{ text: string }]).map(({ text }) => text).join("")
        });

        // Get next response from OpenAI
        const response = await this.openai.responses.create({
          input,
          model: "gpt-4o",
        });

        finalText.push(response.output_text);
      }
    }

    return finalText.join("\n");
  }

  async chatLoop() {
    /**
     * Run an interactive chat loop
     */
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      console.log("\nMCP Client Started!");
      console.log("Type your queries or 'quit' to exit.");

      while (true) {
        const message = await rl.question("\nQuery: ");
        if (message.toLowerCase() === "quit") {
          break;
        }
        const response = await this.processQuery(message);
        console.log("\n" + response);
      }
    } finally {
      rl.close();
    }
  }

  async cleanup() {
    /**
     * Clean up resources
     */
    await this.mcp.close();
  }
}

async function main() {
  if (process.argv.length < 3) {
    console.log("Usage: node build/index.js <path_to_server_script>");
    return;
  }

  const mcpClient = new MCPClient();
  try {
    await mcpClient.connectToServer(process.argv[2]);
    await mcpClient.chatLoop();
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();
