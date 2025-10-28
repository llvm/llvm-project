import lldb
from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider

import inspect, re


class CPythonScriptedFrame(ScriptedFrame):
    """CPython scripted frame with full control over frame behavior."""

    def __init__(self, thread, idx, py_frame_info):
        # Initialize structured data args
        args = lldb.SBStructuredData()
        super().__init__(thread, args)

        self.idx = idx
        self.pc = id(py_frame_info.frame)
        self.function_name = py_frame_info.frame.f_code.co_qualname

        line_entry = lldb.SBLineEntry()
        line_entry.SetFileSpec(lldb.SBFileSpec(py_frame_info.filename, True))
        line_entry.SetLine(py_frame_info.positions.lineno)
        line_entry.SetColumn(py_frame_info.positions.col_offset)

        self.sym_ctx = lldb.SBSymbolContext()
        self.sym_ctx.SetLineEntry(line_entry)

    def get_id(self):
        return self.idx

    def get_pc(self):
        return self.pc

    def get_function_name(self):
        return self.function_name

    def is_artificial(self):
        return False

    def is_hidden(self):
        return False

    def get_symbol_context(self):
        return self.sym_ctx

    def get_register_context(self):
        return None


class CPythonScriptedFrameProvider(ScriptedFrameProvider):
    """Provider that returns ScriptedFrame objects instead of dictionaries."""

    def __init__(self, thread, args):
        super().__init__(thread, args)
        self.python_frames = inspect.stack()

    def get_frame_at_index(self, real_frames, index):
        """Return ScriptedFrame object at given index."""
        if index >= len(self.python_frames):
            return None
        return CPythonScriptedFrame(self.thread, index, self.python_frames[index])
