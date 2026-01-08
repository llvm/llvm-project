import * as fs from "node:fs/promises";
import { Connection } from "./connection";

/**
 * This class is an implementation of the ADB (Android Debug Bridge) protocol.
 * It implements the fundamental building blocks, but it doesn't assemble them
 * into a complete client.
 * For a complete ADB client, see AdbClient.
 */
export class AdbConnection extends Connection {
  async sendAdbMessage(packet: string): Promise<void> {
    const enc = new TextEncoder();
    const data = enc.encode(packet);
    if (data.length > 0xffff) {
      throw new Error(
        "Packet size exceeds maximum allowed length of 65535 bytes",
      );
    }
    const head = enc.encode(data.length.toString(16).padStart(4, "0"));
    await this.write(head);
    await this.write(data);
  }

  async sendDeviceMessage(deviceSerial: string, packet: string): Promise<void> {
    const msg = `host-serial:${deviceSerial}:${packet}`;
    await this.sendAdbMessage(msg);
  }

  async readAdbMessage(): Promise<Uint8Array> {
    const head = await this.read(4);
    if (head.length !== 4) {
      throw new Error("Incomplete ADB message head received");
    }
    const dec = new TextDecoder();
    const size = parseInt(dec.decode(head), 16);
    const message = await this.read(size);
    if (message.length !== size) {
      throw new Error("Incomplete ADB message received");
    }
    return message;
  }

  async readResponseStatus(): Promise<void> {
    const data = await this.read(4);
    if (data.length !== 4) {
      throw new Error("Incomplete ADB message head received");
    }
    const dec = new TextDecoder();
    const status = dec.decode(data);
    switch (status) {
      case "OKAY":
        return;
      case "FAIL":
        const errorMsg = await this.readAdbMessage();
        throw new AdbCommandFailed(
          `ADB command failed: ${dec.decode(errorMsg)}`,
        );
      default:
        throw new Error(`Unknown ADB response status: ${status}`);
    }
  }

  /**
   * Return a list of device serial numbers connected to the ADB server.
   * The ADB server closes the connection after executing this command.
   */
  async getDeviceList(): Promise<string[]> {
    await this.sendAdbMessage("host:devices");
    await this.readResponseStatus();
    const data = await this.readAdbMessage();

    const dec = new TextDecoder();
    const response = dec.decode(data);
    const deviceList: string[] = [];
    const lines = response.split("\n");
    for (const line of lines) {
      const [device] = line.split("\t");
      if (device) {
        deviceList.push(device);
      }
    }
    return deviceList;
  }

  async setTargetDevice(deviceSerial: string): Promise<void> {
    await this.sendAdbMessage(`host:transport:${deviceSerial}`);
    await this.readResponseStatus();
  }

  /**
   * The ADB server closes the connection after executing this command.
   */
  async shellCommandToStream(
    command: string,
    writer: (data: Uint8Array) => Promise<void>,
  ): Promise<void> {
    if (command === "") {
      throw new Error("Shell command cannot be empty");
    }
    const message = `shell:${command}`;
    await this.sendAdbMessage(message);
    await this.readResponseStatus();
    for (;;) {
      const data = await this.read();
      if (data.length === 0) {
        break;
      }
      await writer(data);
    }
  }

  async shellCommandToString(command: string): Promise<string> {
    const output: Uint8Array[] = [];
    const writer = async (data: Uint8Array) => {
      output.push(data);
    };
    await this.shellCommandToStream(command, writer);
    const totalLength = output.reduce((sum, buf) => sum + buf.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const buf of output) {
      combined.set(buf, offset);
      offset += buf.length;
    }
    const dec = new TextDecoder();
    return dec.decode(combined);
  }

  async shellCommand(command: string): Promise<void> {
    const writer = async (data: Uint8Array) => {};
    await this.shellCommandToStream(command, writer);
  }

  /**
   * The ADB server closes the connection after executing this command.
   */
  async getPid(packageName: string): Promise<number> {
    const output = await this.shellCommandToString(`pidof ${packageName}`);
    const pid = parseInt(output.trim(), 10);
    if (isNaN(pid)) {
      throw new AdbCommandFailed(
        `Failed to get PID for package: ${packageName}`,
      );
    }
    return pid;
  }

  async addPortForwarding(
    deviceSerial: string,
    remotePort: number | string,
    localPort: number = 0,
  ): Promise<number> {
    if (typeof remotePort === "number") {
      remotePort = `tcp:${remotePort}`;
    }
    const message = `forward:tcp:${localPort};${remotePort}`;
    await this.sendDeviceMessage(deviceSerial, message);
    await this.readResponseStatus();
    await this.readResponseStatus();
    const result = await this.readAdbMessage();
    const dec = new TextDecoder();
    const port = parseInt(dec.decode(result), 10);
    if (isNaN(port)) {
      throw new Error("Failed to add port forwarding");
    }
    return port;
  }

  async removePortForwarding(
    deviceSerial: string,
    localPort: number,
  ): Promise<void> {
    const message = `killforward:tcp:${localPort}`;
    await this.sendDeviceMessage(deviceSerial, message);
    await this.readResponseStatus();
  }

  async getPortForwardingList(): Promise<
    { device: string; localPort: string; remotePort: string }[]
  > {
    const message = `host:list-forward`;
    await this.sendAdbMessage(message);
    await this.readResponseStatus();
    const result = await this.readAdbMessage();
    const dec = new TextDecoder();
    const list = dec.decode(result);
    const ret: { device: string; localPort: string; remotePort: string }[] = [];
    for (const line of list.split("\n")) {
      const elems = line.split(" ");
      if (elems.length === 3) {
        ret.push({
          device: elems[0],
          localPort: elems[1],
          remotePort: elems[2],
        });
      }
    }
    return ret;
  }

  async enterSyncMode(): Promise<void> {
    await this.sendAdbMessage("sync:");
    await this.readResponseStatus();
  }

  async writeSyncHeader(requestId: string, dataLen: number): Promise<void> {
    const enc = new TextEncoder();
    const requestIdData = enc.encode(requestId);
    if (requestIdData.length !== 4) {
      throw new Error("Sync request ID must be 4 characters long");
    }
    const buf = new Uint8Array(8);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    buf.set(requestIdData, 0);
    view.setUint32(4, dataLen, true);
    await this.write(buf);
  }

  async writeSyncData(requestId: string, data: Uint8Array): Promise<void> {
    await this.writeSyncHeader(requestId, data.length);
    await this.write(data);
  }

  async readSyncHeader(): Promise<{ responseId: string; dataLen: number }> {
    const header = await this.read(8);
    if (header.length !== 8) {
      throw new Error("Incomplete sync header received");
    }
    const view = new DataView(
      header.buffer,
      header.byteOffset,
      header.byteLength,
    );
    const dec = new TextDecoder();
    const responseId = dec.decode(header.subarray(0, 4));
    const dataLen = view.getUint32(4, true);
    return { responseId, dataLen };
  }

  async pushStream(
    reader: (maxSize: number) => Promise<Uint8Array>,
    remoteFilePath: string,
  ): Promise<void> {
    const defaultMode = 0o100770;
    const maxChunkSize = 2 * 1024;

    if (remoteFilePath.indexOf(",") !== -1) {
      throw new Error("Remote file path cannot contain commas");
    }

    const enc = new TextEncoder();
    const body = `${remoteFilePath},0${defaultMode.toString(8)}`;
    await this.writeSyncData("SEND", enc.encode(body));

    for (;;) {
      const chunk = await reader(maxChunkSize);
      if (chunk.length === 0) {
        break;
      }
      await this.writeSyncData("DATA", chunk);
    }

    const mtime = Math.floor(Date.now() / 1000);
    await this.writeSyncHeader("DONE", mtime); // TODO: year 2038 bug

    const { responseId, dataLen } = await this.readSyncHeader();
    if (responseId === "FAIL") {
      const errorMessageData = await this.read(dataLen);
      const dec = new TextDecoder();
      const errorMessage = dec.decode(errorMessageData);
      throw new AdbCommandFailed(
        `Failed to push file "${remoteFilePath}": ${errorMessage}`,
      );
    } else if (responseId !== "OKAY") {
      throw new Error(`Unexpected sync response: ${responseId}`);
    }
    if (dataLen !== 0) {
      throw new Error("Unexpected data in sync OKAY response");
    }
  }

  async pushData(data: Uint8Array, remoteFilePath: string): Promise<void> {
    let offset = 0;
    const reader = async (maxSize: number): Promise<Uint8Array> => {
      const chunk = data.subarray(offset, offset + maxSize);
      offset += chunk.length;
      return chunk;
    };
    await this.pushStream(reader, remoteFilePath);
  }

  async pushFile(localFilePath: string, remoteFilePath: string): Promise<void> {
    const handle = await fs.open(localFilePath, "r");
    let buf = new Uint8Array(0);
    const reader = async (maxSize: number): Promise<Uint8Array> => {
      if (buf.length < maxSize) {
        buf = new Uint8Array(maxSize);
      }
      const { bytesRead } = await handle.read(buf, 0, maxSize, null);
      return buf.subarray(0, bytesRead);
    };
    await this.pushStream(reader, remoteFilePath);
  }
}

export class AdbCommandFailed extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdbCommandFailed";
  }
}
