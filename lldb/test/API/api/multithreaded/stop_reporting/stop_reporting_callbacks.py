import lldb
import time

signal_var = False
failed_calls = 0

# Long enough for the driver's other thread to finish its checks, short enough
# that a broken test fails instead of hanging.
DEADLINE_SECONDS = 60


def slow_breakpoint_callback(frame, bp_loc, extra_args, unused):
    global signal_var, failed_calls
    # Keep calling a function in the target until the driver's other thread
    # says to stop, so that thread has time to check what it sees meanwhile.
    signal_var = True
    options = lldb.SBExpressionOptions()
    options.SetStopOthers(True)
    options.SetTryAllThreads(False)

    deadline = time.monotonic() + DEADLINE_SECONDS
    while signal_var and time.monotonic() < deadline:
        time.sleep(0.01)
        valobj = frame.EvaluateExpression("call_me_function(1)", options)
        if not valobj.error.success:
            failed_calls += 1


def handle_signal_var(debugger, command, result, unused):
    global signal_var
    if len(command) == 0:
        result.Print("True" if signal_var else "False")
        result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
    else:
        signal_var = False
        result.SetStatus(lldb.eReturnStatusSuccessFinishNoResult)


def handle_failed_calls(debugger, command, result, unused):
    result.Print(str(failed_calls))
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
