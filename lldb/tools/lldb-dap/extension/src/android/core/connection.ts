import * as net from 'node:net';

/**
 * This class is a TCP client.
 */
export class Connection {

    private socket: net.Socket | null = null;
    private rxBuffer: Buffer[] = [];
    private rxSignal: Signal | null = null;
    private rxEndOfStream = false;
    private inError: Error | null = null;

    async connect(host: string, port: number) {
        if (this.socket) {
            throw new Error('Connection is already established');
        }
        const socket = await connectSocket(host, port);
        this.socket = socket;

        socket.on('data', (data: Buffer) => {
            this.rxBuffer.push(data);
            if (this.rxSignal) {
                this.rxSignal.fire();
                this.rxSignal = null;
            }
        });
        socket.on('end', () => {
            // the peer has performed a tx shutdown or closed the connection
            this.rxEndOfStream = true;
            if (this.rxSignal) {
                this.rxSignal.fire();
                this.rxSignal = null;
            }
        });
        socket.on('error', (err: Error) => {
            console.error(`Socket error: ${err.message}`);
            this.inError = err;
            this.socket?.destroy();
            this.socket = null;
            this.rxBuffer = [];
            if (this.rxSignal) {
                this.rxSignal.fire();
                this.rxSignal = null;
            }
            this.rxEndOfStream = false;
        });
    }

    /**
     * Signal the end of data transmission. Reception of data may still continue.
     */
    async end(): Promise<void> {
        if (this.inError) {
            const err = this.inError;
            this.inError = null;
            throw err;
        }
        const socket = this.socket;
        if (!socket) {
            throw new Error('Connection is closed');
        }
        return new Promise((resolve, reject) => {
            socket.end(() => {
                resolve();
            });
        });
    }

    /**
     * Close the connection immediately.
     * This discards buffered data.
     */
    close(): void {
        const socket = this.socket;
        if (socket) {
            socket.destroy();
            this.socket = null;
            this.rxBuffer = [];
            if (this.rxSignal) {
                this.rxSignal.fire();
                this.rxSignal = null;
            }
            this.rxEndOfStream = false;
        }
    }

    get isConnected(): boolean {
        return this.socket !== null;
    }

    async write(data: Uint8Array): Promise<void> {
        if (this.inError) {
            const err = this.inError;
            this.inError = null;
            throw err;
        }
        const socket = this.socket;
        if (!socket) {
            throw new Error('Connection is closed');
        }
        if (data.length === 0) {
            return;
        }
        return new Promise((resolve, reject) => {
            socket.write(data, (err) => {
                if (err) {
                    return reject(err);
                }
                resolve();
            });
        });
    }

    /**
     * Get the number of bytes currently available in the receive buffer.
     */
    get availableData(): number {
        let totalLength = 0;
        for (const buf of this.rxBuffer) {
            totalLength += buf.length;
        }
        return totalLength;
    }

    /**
     * Read `size` bytes from the receive buffer. If `size` is undefined, read
     * all available data, but at least one byte.
     * If the requested data is not yet available, wait until it is received.
     * If the end of the stream is reached before the requested data is
     * available, returns whatever is available (may be zero bytes).
     */
    async read(size?: number): Promise<Uint8Array> {
        return new Promise(async (resolve, reject) => {
            for (;;) {
                if (this.inError) {
                    const err = this.inError;
                    this.inError = null;
                    reject(err);
                    return;
                }
                if (!this.socket) {
                    reject(new Error('Connection is closed'));
                    return;
                }
                if (this.rxSignal) {
                    reject(new Error('Concurrent read operations are not supported'));
                    return;
                }
                const available = this.availableData;
                if (available >= (size ?? 1)) {
                    const buffer = this.readFromRxBuffer(size ?? available);
                    resolve(buffer);
                    return;
                }
                if (this.rxEndOfStream) {
                    const buffer = this.readFromRxBuffer(available);
                    resolve(buffer);
                    return;
                }
                this.rxSignal = new Signal();
                await this.rxSignal.promise;
            }
        });
    }

    private readFromRxBuffer(size: number): Uint8Array {
        const buffer = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
            if (this.rxBuffer.length === 0) {
                throw new Error('Not enough data in rxBuffer');
            }
            const chunk = this.rxBuffer[0];
            const toCopy = Math.min(size - offset, chunk.length);
            buffer.set(chunk.subarray(0, toCopy), offset);
            offset += toCopy;
            if (toCopy < chunk.length) {
                this.rxBuffer[0] = chunk.subarray(toCopy);
            } else {
                this.rxBuffer.shift();
            }
        }
        return buffer;
    }
}

class Signal {
    private _resolve!: () => void;
    private _reject!: (error: Error) => void;

    readonly promise: Promise<void>;

    constructor() {
        this.promise = new Promise<void>((res, rej) => {
            this._resolve = res;
            this._reject = rej;
        });
    }

    fire() {
        this._resolve();
    }

    fireError(error: Error) {
        this._reject(error);
    }
}

async function connectSocket(host: string, port: number): Promise<net.Socket> {
    return new Promise((resolve, reject) => {
        let socket: net.Socket;

        const errorHandler = (err: Error) => {
            console.error(`Connection error: ${err.message}`);
            reject(err);
        };

        const endHandler = () => {
            // can this happen?
            errorHandler(new Error('Connection ended unexpectedly'));
        };

        socket = net.createConnection({ host, port }, () => {
            socket.off('error', errorHandler);
            socket.off('end', endHandler);
            resolve(socket);
        });

        socket.on('error', errorHandler);
        socket.on('end', endHandler);
    });
}
