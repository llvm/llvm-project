import * as vscode from "vscode";
import * as winston from "winston";
import * as Transport from "winston-transport";

class OutputChannelTransport extends Transport {
    constructor(private readonly ouptutChannel: vscode.OutputChannel) {
        super();
    }

    public log(info: any, next: () => void): void {
        this.ouptutChannel.appendLine(info[Symbol.for('message')]);
        next();
    }
}

export class Logger implements vscode.Disposable {
    private disposables: vscode.Disposable[] = [];
    private logger: winston.Logger;

    constructor(public readonly logFilePath: (name: string) => string, ouptutChannel: vscode.OutputChannel) {
        const ouptutChannelTransport = new OutputChannelTransport(ouptutChannel);
        ouptutChannelTransport.level = this.outputChannelLevel();
        this.logger = winston.createLogger({
            transports: [
                new winston.transports.File({ filename: logFilePath("lldb-dap-extension.log"), level: "debug" }), // File logging at the 'debug' level
                ouptutChannelTransport
            ],
            format: winston.format.combine(
                winston.format.errors({ stack: true }),
                winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
                winston.format.printf(msg => `[${msg.timestamp}][${msg.level}] ${msg.message} ${msg.stack ? msg.stack : ''}`),
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

    debug(message: any) {
        this.logger.debug(this.normalizeMessage(message));
    }

    info(message: any) {
        this.logger.info(this.normalizeMessage(message));
    }

    warn(message: any) {
        this.logger.warn(this.normalizeMessage(message));
    }

    error(message: any) {
        if (message instanceof Error) {
            this.logger.error(message);
            return;
        }
        this.logger.error(this.normalizeMessage(message));
    }

    private normalizeMessage(message: any) {
        if (typeof message === "string") {
            return message;
        } else if (typeof message === "object") {
            return JSON.stringify(message);
        }
        return `${message}`;
    }

    private outputChannelLevel(): string {
        return vscode.workspace.getConfiguration("lldb-dap").get("verboseLogging", false) ?
            "debug" : "info";
    }

    dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}