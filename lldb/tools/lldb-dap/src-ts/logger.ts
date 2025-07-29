import * as vscode from "vscode";
import * as winston from "winston";
import * as TransportType from "winston-transport";

// Runtime error if don't use "require"
const Transport: typeof TransportType = require("winston-transport");

class OutputChannelTransport extends Transport {
    constructor(private readonly ouptutChannel: vscode.OutputChannel) {
        super();
    }

    public log(info: any, next: () => void): void {
        this.ouptutChannel.appendLine(info[Symbol.for('message')]);
        next();
    }
}

export type LogFilePathProvider = (name: string) => string;

export interface Logger {
  debug(message: string, ...args: any[]): void
  error(error: string | Error, ...args: any[]): void
  info(message: string, ...args: any[]): void
  warn(message: string, ...args: any[]): void
}

export class LLDBDAPLogger implements vscode.Disposable {
    private disposables: vscode.Disposable[] = [];
    private logger: winston.Logger;

    constructor(public readonly logFilePath: string, ouptutChannel: vscode.OutputChannel) {
        const ouptutChannelTransport = new OutputChannelTransport(ouptutChannel);
        ouptutChannelTransport.level = this.outputChannelLevel();
        this.logger = winston.createLogger({
            transports: [
                new winston.transports.File({ filename: logFilePath, level: "debug" }), // File logging at the 'debug' level
                ouptutChannelTransport
            ],
            format: winston.format.combine(
                winston.format.errors({ stack: true }),
                winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" }), // This is the format of `vscode.LogOutputChannel`
                winston.format.printf(msg => `${msg.timestamp} [${msg.level}] ${msg.message} ${msg.stack ? msg.stack : ''}`),
            ),
        });
        if (process.env.NODE_ENV !== 'production') {
            this.logger.add(new winston.transports.Console({
                level: "error"
            }));
        }
        this.disposables.push(
            {
                dispose: () => this.logger.close()
            },
            vscode.workspace.onDidChangeConfiguration(e => {
                if (e.affectsConfiguration("lldb-dap.verboseLogging")) {
                    ouptutChannelTransport.level = this.outputChannelLevel();
                }
            })
        );
    }

    debug(message: string, ...args: any[]) {
        this.logger.debug([message, ...args].map(m => this.normalizeMessage(m)).join(" "));
    }

    info(message: string, ...args: any[]) {
        this.logger.info([message, ...args].map(m => this.normalizeMessage(m)).join(" "));
    }

    warn(message: string, ...args: any[]) {
        this.logger.warn([message, ...args].map(m => this.normalizeMessage(m)).join(" "));
    }

    error(message: Error | string, ...args: any[]) {
        if (message instanceof Error) {
            this.logger.error(message);
            this.logger.error([...args].map(m => this.normalizeMessage(m)).join(" "));
            return;
        }
        this.logger.error([message, ...args].map(m => this.normalizeMessage(m)).join(" "));
    }

    private normalizeMessage(message: any) {
        if (typeof message === "string") {
            return message;
        }
        try {
            return JSON.stringify(message);
        } catch (e) {
            return `${message}`;
        }
    }

    private outputChannelLevel(): string {
        return vscode.workspace.getConfiguration("lldb-dap").get("verboseLogging", false) ?
            "debug" : "info";
    }

    dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}