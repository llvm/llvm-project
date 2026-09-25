import struct, signal

from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread

# A pc with junk in the bits above the addressable ones. Bit 55 is clear, so
# the ABI strips the non-addressable bits instead of sign-extending them.
TAGGED_PC = 0xD169800100004000
ADDRESSABLE_BITS = 34
FIXED_PC = 0x0000000100004000


class AddressableBitsScriptedProcess(ScriptedProcess):
    def __init__(self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.threads[0] = AddressableBitsScriptedThread(self, args)
        self.threads[1] = ScriptedFramesScriptedThread(self, args)
        self.addressable_bits = {"lowmem": ADDRESSABLE_BITS}

    def read_memory_at_address(
        self, addr: int, size: int, error: lldb.SBError
    ) -> lldb.SBData:
        data = lldb.SBData()
        # The buffer has to outlive this call, so the SBData needs to own it
        # rather than alias a Python object we're about to drop.
        data.SetDataWithOwnership(
            error,
            bytes(size),
            self.target.GetByteOrder(),
            self.target.GetAddressByteSize(),
        )
        return data

    def get_loaded_images(self) -> list:
        return self.loaded_images

    def get_process_id(self) -> int:
        return 42

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self) -> str:
        return (
            AddressableBitsScriptedThread.__module__
            + "."
            + AddressableBitsScriptedThread.__name__
        )


class AddressableBitsScriptedThread(ScriptedThread):
    REGISTER_PC = TAGGED_PC

    def __init__(self, process, args):
        super().__init__(process, args)

    def get_thread_id(self) -> int:
        return 0x19

    def get_state(self) -> int:
        return lldb.eStateStopped

    def get_stop_reason(self) -> Dict[str, Any]:
        return {"type": lldb.eStopReasonSignal, "data": {"signal": signal.SIGINT}}

    def get_register_context(self) -> str:
        # The unwinder needs a plausible stack pointer to build frame zero.
        regs = [0] * 33
        regs[29] = 0x16FDFF000  # fp
        regs[30] = self.REGISTER_PC  # lr
        regs[31] = 0x16FDFE000  # sp
        regs[32] = self.REGISTER_PC  # pc
        return struct.pack(f"{len(regs)}Q", *regs) + struct.pack("I", 0)


class ScriptedFramesScriptedThread(AddressableBitsScriptedThread):
    # An address that needs no fixing, so that asserting on FIXED_PC can only
    # pass if frame zero really came from get_stackframes() below.
    REGISTER_PC = 0x16FDFD000

    def get_thread_id(self) -> int:
        return 0x1A

    def get_stackframes(self) -> list:
        return [{"pc": TAGGED_PC}]
