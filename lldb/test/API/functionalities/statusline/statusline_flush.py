"""Helper for TestStatusline.test_flush_deadlock.

Flushes the process while the event thread redraws the statusline and waits on
the target API mutex.
"""

import time

import lldb


@lldb.command("statusline_flush")
def statusline_flush(debugger, command, result, internal_dict):
    # SBDebugger.HandleCommand holds the target API mutex for the duration of
    # the nested command.
    debugger.HandleCommand("statusline_flush_locked " + command)


@lldb.command("statusline_flush_locked")
def statusline_flush_locked(debugger, command, result, internal_dict):
    # The progress event makes the event thread redraw the statusline, which
    # waits on the target API mutex held by the caller.
    lldb.SBProgress("flush", "redraw", debugger).Finalize()
    time.sleep(0.5)
    # Changing a load address flushes the process.
    debugger.HandleCommand("target modules load -f a.out -s " + command)
