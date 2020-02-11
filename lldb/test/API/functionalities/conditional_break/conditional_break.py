import lldb


def stop_if_called_from_a(frame, bp_loc, dict):

    thread = frame.GetThread()
    process = thread.GetProcess()
    target = process.GetTarget()
    dbg = target.GetDebugger()

    # Perform synchronous interaction with the debugger.
    old_async = dbg.GetAsync()
    dbg.SetAsync(True)

    # We check the call frames in order to stop only when the immediate caller
    # of the leaf function c() is a().  If it's not the right caller, we ask the
    # command interpreter to continue execution.

    should_stop = True
    if thread.GetNumFrames() >= 2:

        if (thread.frames[0].function.name ==
                'c' and thread.frames[1].function.name == 'a'):
            should_stop = True
        else:
            should_stop = False

    dbg.SetAsync(old_async)
    return should_stop
