import lldb

# The records live on the lldb module, which isn't reloaded with this script.


class ReadHeader:
    """A one-shot stop hook that reads the image's Mach-O header from the
    process, which can't be read when the image is added to the target."""

    def __init__(self, target, extra_args, internal_dict):
        pass

    def handle_stop(self, exe_ctx, stream):
        target = exe_ctx.GetTarget()
        header = target.module["libpayload.dylib"].GetObjectFileHeaderAddress()
        error = lldb.SBError()
        magic = exe_ctx.GetProcess().ReadUnsignedFromMemory(
            header.GetLoadAddress(target), 4, error
        )
        index = target.GetDebugger().GetIndexOfTarget(target)
        read = magic if error.Success() else error.GetCString()
        lldb.payload_magics = getattr(lldb, "payload_magics", []) + [(index, read)]
        return True


def __lldb_init_module(debugger, internal_dict):
    lldb.payload_inits = getattr(lldb, "payload_inits", 0) + 1


def __lldb_module_added_to_target(target, internal_dict):
    index = target.GetDebugger().GetIndexOfTarget(target)
    lldb.payload_targets = getattr(lldb, "payload_targets", []) + [index]

    # The scripted process only stops when it launches, which is the initial
    # stop, so the hook asks to run there.
    result = lldb.SBCommandReturnObject()
    target.GetDebugger().GetCommandInterpreter().HandleCommand(
        f"target stop-hook add --one-shot true --at-initial-stop true "
        f"-P {__name__}.ReadHeader",
        lldb.SBExecutionContext(target),
        result,
    )
