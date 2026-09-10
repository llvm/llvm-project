"""
Frame provider whose scripted frame replaces a native frame and forwards that
frame's registers through get_register_context.

Reporting a register context is what makes LLDB evaluate the frame's DWARF
location expressions when it formats variables.
"""

import struct

import lldb
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider
from lldb.plugins.scripted_process import ScriptedFrame

WRAPPED_FUNCTION = "compute"


class WrappedFrame(ScriptedFrame):
    def __init__(self, thread, frame, idx):
        super().__init__(thread, lldb.SBStructuredData())
        self._frame = frame
        self._idx = idx

    def get_id(self):
        return self._idx

    def get_pc(self):
        return self._frame.GetPC()

    def get_symbol_context(self):
        return self._frame.GetSymbolContext(lldb.eSymbolContextEverything)

    def get_function_name(self):
        return self._frame.GetFunctionName() or "<wrapped>"

    def is_artificial(self):
        return False

    def is_hidden(self):
        return False

    def get_register_context(self):
        """Forward the wrapped frame's GPRs, packed in register_info order."""
        regs = {}
        for reg_set in self._frame.registers:
            if "general purpose" in reg_set.name.lower():
                for reg in reg_set:
                    regs[reg.name] = int(reg.value, 16) if reg.value else 0
                break
        if not regs:
            return None

        info = self.get_register_info()["registers"]

        def read(entry):
            # A register set reports a register under the name LLDB displays,
            # which can be an alias of the architectural name the register info
            # uses. The register info carries that alias in "alt-name".
            if entry["name"] in regs:
                return regs[entry["name"]]
            return regs.get(entry.get("alt-name", ""), 0)

        return struct.pack(f"{len(info)}Q", *(read(r) for r in info))


class WrapVariablesProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return f"Wrap the {WRAPPED_FUNCTION!r} frame and forward its registers"

    def get_frame_at_index(self, index):
        if index >= self.input_frames.GetSize():
            return None

        frame = self.input_frames.GetFrameAtIndex(index)
        if not frame.IsValid():
            return None

        if frame.GetFunctionName() == WRAPPED_FUNCTION:
            return WrappedFrame(self.thread, frame, index)

        # Returning an int reuses the input frame at that index unchanged.
        return index
