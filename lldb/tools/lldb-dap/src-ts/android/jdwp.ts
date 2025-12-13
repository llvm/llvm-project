import { Connection } from "./connection";

/*
 * Protocol specs: https://docs.oracle.com/javase/8/docs/technotes/guides/jpda/jdwp-spec.html
 */

namespace Jdwp {

    enum CommandFlag {
        REPLY = 0x80,
    }

    enum CommandSet {
        VM = 1,
        CLASS = 3,
        OBJECT = 9,
    }

    enum VmCommand {
        VERSION = 1,
    }

    type CommandPacket = {
        id: number;
        flags: number;
        commandSet: CommandSet;
        command: number;
        data: Uint8Array;
    }

    type ReplayPacket = {
        id: number;
        flags: number;
        errorCode: number;
        data: Uint8Array;
    }

    function encodePacket(command: CommandPacket): Uint8Array {
        const packet = new Uint8Array(11 + command.data.length);
        const view = new DataView(packet.buffer, packet.byteOffset, packet.byteLength);
        view.setUint32(0, 11 + command.data.length, false); // length
        view.setUint32(4, command.id, false); // id
        view.setUint8(8, command.flags); // flags
        view.setUint8(9, command.commandSet); // commandSet
        view.setUint8(10, command.command); // command
        packet.set(command.data, 11);
        return packet;
    }

    function decodePacket(buffer: Uint8Array): ReplayPacket {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        const length = view.getUint32(0, false);
        const id = view.getUint32(4, false);
        const flags = view.getUint8(8);
        const errorCode = view.getUint16(9, false);
        const data = buffer.subarray(11, length);
        return {
            id,
            flags,
            errorCode,
            data,
        };
    }

    function encodeVersionCommand() {
        const command: CommandPacket = {
            id: 1,
            flags: 0,
            commandSet: CommandSet.VM,
            command: VmCommand.VERSION,
            data: new Uint8Array(0),
        };
        return encodePacket(command);
    }

    function decodeVersionReply(buffer: Uint8Array) {
        const packet = decodePacket(buffer);
        if ((packet.flags & CommandFlag.REPLY) === 0) {
            throw new Error('Not a reply packet');
        }
        if (packet.errorCode !== 0) {
            throw new Error(`JDWP error code: ${packet.errorCode}`);
        }
        const view = new DataView(packet.data.buffer, packet.data.byteOffset, packet.data.byteLength);
        const dec = new TextDecoder();
        let offset = 0;
        const descLength = view.getUint32(offset, false);
        offset += 4;
        const desc = dec.decode(packet.data.subarray(offset, offset + descLength));
        offset += descLength;
        const jdwpMajor = view.getUint32(offset, false);
        offset += 4;
        const jdwpMinor = view.getUint32(offset, false);
        offset += 4;
        const vmVersionLength = view.getUint32(offset, false);
        offset += 4;
        const vmVersion = dec.decode(packet.data.subarray(offset, offset + vmVersionLength));
        offset += vmVersionLength;
        const vmNameLength = view.getUint32(offset, false);
        offset += 4;
        const vmName = dec.decode(packet.data.subarray(offset, offset + vmNameLength));
        offset += vmNameLength;
        if (offset !== packet.data.length) {
            throw new Error('Unexpected data in version reply packet');
        }
        return {
            description: desc,
            jdwpMajor,
            jdwpMinor,
            vmVersion,
            vmName,
        };
    }

    async function readPacket(c: Connection): Promise<Uint8Array> {
        const head = await c.read(11);
        if (head.length !== 11) {
            throw new Error('Incomplete JDWP packet header received');
        }
        const view = new DataView(head.buffer);
        const length = view.getUint32(0, false);
        const body = await c.read(length - 11);
        if (body.length !== length - 11) {
            throw new Error('Incomplete JDWP packet body received');
        }
        const full = new Uint8Array(length);
        full.set(head, 0);
        full.set(body, 11);
        return full;
    }

    export async function handshake(c: Connection) {
        c.write(Buffer.from("JDWP-Handshake"));
        const reply = await c.read(14);
        const dec = new TextDecoder();
        if (dec.decode(reply) !== "JDWP-Handshake") {
            throw new Error('Invalid JDWP handshake reply');
        }
    }

    export async function getVersion(c: Connection) {
        const versionCommand = encodeVersionCommand();
        await c.write(versionCommand);
        const replyBuffer = await readPacket(c);
        const versionReply = decodeVersionReply(replyBuffer);
        return versionReply;
    }
}

export default Jdwp;
