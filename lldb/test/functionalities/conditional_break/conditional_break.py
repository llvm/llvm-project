import sys
import lldb
import lldbutil

def stop_if_called_from_a():
    # lldb.debugger_unique_id stores the id of the debugger associated with us.
    dbg = lldb.SBDebugger.FindDebuggerWithID(lldb.debugger_unique_id)

    # Perform synchronous interaction with the debugger.
    old_async = dbg.GetAsync()
    dbg.SetAsync(True)

    # Retrieve the target, process, and the only thread.
    target = dbg.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetThreadAtIndex(0)

    # We check the call frames in order to stop only when the immediate caller
    # of the leaf function c() is a().  If it's not the right caller, we ask the
    # command interpreter to continue execution.

    print >> sys.stdout, "Checking call frames..."
    lldbutil.print_stacktrace(thread)
    should_stop = True
    if thread.GetNumFrames() >= 2:
        funcs = lldbutil.get_function_names(thread)
        print >> sys.stdout, funcs[0], "called from", funcs[1]
        if (funcs[0] == 'c' and funcs[1] == 'a'):
            should_stop = True
        else:
            process.Continue()
            should_stop = False

    dbg.SetAsync(old_async)
    return should_stop


