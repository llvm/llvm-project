import lldb


def report_command(debugger, command, exe_ctx, result, internal_dict):
    global stop_thread
    print(f"About to report out stop_thread: {stop_thread}")
    mssg = f"Stop Threads: {stop_thread}"
    result.AppendMessage(mssg)

    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)


class stop_handler:
    def __init__(self, target, extra_args, dict):
        global stop_thread
        stop_thead = 0
        self.target = target

    def handle_stop(self, exe_ctx, stream):
        global stop_thread
        thread = exe_ctx.thread
        stop_thread = thread.idx


def __lldb_init_module(debugger, internal_dict):
    global stop_thread
    stop_thread = 0
    debugger.HandleCommand(
        f"command script add -o -f '{__name__}.report_command' report_command"
    )
