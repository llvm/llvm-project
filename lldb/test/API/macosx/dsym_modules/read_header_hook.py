import lldb

# __lldb_module_added_to_target can't read the process: the module may not
# have a load address yet, and there may be no process. It adds a one-shot
# stop hook instead, which reads the module's Mach-O header at the next stop.


class ReadHeader:
    def __init__(self, target, extra_args, internal_dict):
        pass

    def handle_stop(self, exe_ctx, stream):
        lldb.read_header_runs = getattr(lldb, "read_header_runs", 0) + 1
        target = exe_ctx.GetTarget()
        module = target.FindModule(target.GetExecutable())
        header = module.GetObjectFileHeaderAddress().GetLoadAddress(target)
        if header == lldb.LLDB_INVALID_ADDRESS:
            # Too early, for example the initial stop of a launch, before the
            # loader has run: try again at the next stop.
            add_hook(target)
            return True
        error = lldb.SBError()
        magic = exe_ctx.GetProcess().ReadUnsignedFromMemory(header, 4, error)
        lldb.read_header_magic = magic if error.Success() else error.GetCString()
        return True


def add_hook(target):
    result = lldb.SBCommandReturnObject()
    target.GetDebugger().GetCommandInterpreter().HandleCommand(
        f"target stop-hook add --one-shot true -P {__name__}.ReadHeader",
        lldb.SBExecutionContext(target),
        result,
    )


def __lldb_module_added_to_target(target, internal_dict):
    add_hook(target)
