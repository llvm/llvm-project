import lldb


def set_globals(target, index, value):
    exe_module = target.FindModule(target.executable)
    var = exe_module.FindFirstGlobalVariable(target, "g_global")
    child = var.GetChildAtIndex(index)
    child.SetValueFromCString(str(value))


def outer_callback(frame: lldb.SBFrame, bp_loc, internal_dict):
    thread = frame.GetThread()

    # address of the next frame
    next_frame_pc = thread.get_thread_frames()[1].GetPC()

    target = thread.process.target
    bp = target.BreakpointCreateByAddress(next_frame_pc)
    bp.SetScriptCallbackFunction(f"{__name__}.nested_bp_callback")
    set_globals(target, 1, 1)
    set_globals(target, 2, bp.GetID())

    return False


def nested_bp_callback(frame: lldb.SBFrame, bp_loc, extra_args, internal_dict):
    target = frame.thread.process.target
    set_globals(target, 1, 2)

    return True
