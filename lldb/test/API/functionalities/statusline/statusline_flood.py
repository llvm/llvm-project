"""Helper for TestStatusline.test_scripted_command_output_not_eaten.

Registers a command that floods output while emitting progress events, so the
statusline redraws (on the event thread) concurrently with the command output.
"""

import lldb


@lldb.command("statusline_flood")
def statusline_flood(debugger, command, result, internal_dict):
    # The caller sets the count so the test can widen the flood and give the
    # event thread a chance to redraw the statusline mid-output. The loop never
    # sleeps: a redraw must be able to land between prints for a race to show.
    count = int(command) if command.strip() else 50
    progress = lldb.SBProgress("flood", "working", count, debugger)
    for i in range(count):
        print("MARKER_{:04d}".format(i), flush=True)
        progress.Increment(1, "step {}".format(i))
    progress.Finalize()
