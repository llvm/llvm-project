import lldb
from lldb.plugins.scripted_process import ScriptedProcess, ScriptedThread


class ImageScriptedProcess(ScriptedProcess):
    """A process with one stopped thread and one loaded image, whose memory is
    the image's file mapped at the load address it's given."""

    def __init__(self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.path = args.GetValueForKey("path").GetStringValue(4096)
        self.load_addr = args.GetValueForKey("load_addr").GetUnsignedIntegerValue()
        with open(self.path, "rb") as image:
            self.image = image.read()
        self.loaded_images.append({"path": self.path, "load_addr": self.load_addr})
        self.threads[0] = ImageScriptedThread(self, args)

    def read_memory_at_address(
        self, addr: int, size: int, error: lldb.SBError
    ) -> lldb.SBData:
        data = lldb.SBData()
        offset = addr - self.load_addr
        if offset < 0 or offset + size > len(self.image):
            error.SetErrorString(f"0x{addr:x} isn't in {self.path}")
            return data
        data.SetDataWithOwnership(
            error,
            self.image[offset : offset + size],
            self.target.GetByteOrder(),
            self.target.GetAddressByteSize(),
        )
        return data

    def get_loaded_images(self):
        return self.loaded_images

    def get_process_id(self) -> int:
        return 42

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self) -> str:
        return f"{__name__}.{ImageScriptedThread.__name__}"


class ImageScriptedThread(ScriptedThread):
    def __init__(self, process, args):
        super().__init__(process, args)

    def get_thread_id(self) -> int:
        return 1

    def get_name(self) -> str:
        return ImageScriptedThread.__name__

    def get_state(self) -> int:
        return lldb.eStateStopped

    def get_stop_reason(self):
        return {"type": lldb.eStopReasonTrace, "data": {}}

    def get_register_context(self) -> bytes:
        # The test never reads the registers, so each one is zero.
        context = b""
        for register in self.get_register_info()["registers"]:
            context += bytes(register["bitsize"] // 8)
        return context
