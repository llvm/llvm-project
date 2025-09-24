import lldb


class stop_handler:
    def __init__(self, target, extra_args, dict):
        self.extra_args = extra_args
        self.target = target
        self.counter = 0
        ret_val = self.extra_args.GetValueForKey("return_false")
        if ret_val:
            self.ret_val = False
        else:
            self.ret_val = True

    def handle_stop(self, exe_ctx, stream):
        self.counter += 1
        stream.Print("I have stopped %d times.\n" % (self.counter))
        increment = 1
        value = self.extra_args.GetValueForKey("increment")
        if value:
            increment = value.GetUnsignedIntegerValue()
        else:
            stream.Print("Could not find increment in extra_args\n")
        frame = exe_ctx.GetFrame()
        expression = "g_var += %d" % (increment)
        expr_result = frame.EvaluateExpression(expression)
        if not expr_result.GetError().Success():
            stream.Print(
                "Error running expression: %s" % (expr_result.GetError().GetCString())
            )
        value = exe_ctx.target.FindFirstGlobalVariable("g_var")
        if not value.IsValid():
            stream.Print("Didn't get a valid value for g_var.")
        else:
            int_val = value.GetValueAsUnsigned()
        stream.Print("Returning value: %d from handle_stop.\n" % (self.ret_val))
        return self.ret_val


class bad_handle_stop:
    def __init__(self, target, extra_args, dict):
        print("I am okay")

    def handle_stop(self):
        print("I am bad")


class no_handle_stop:
    def __init__(self, target, extra_args, dict):
        print("I am okay")


class self_deleting_stop:
    def __init__(self, target, extra_args, dict):
        self.target = target

    def handle_stop(self, exe_ctx, stream):
        interp = exe_ctx.target.debugger.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("handle_id", result)
        id_str = result.GetOutput().rstrip()

        command = f"target stop-hook delete {id_str}"
        interp.HandleCommand(command, result)


stop_hook_id = 0


def handle_stop_hook_id(debugger, command, exe_ctx, result, extra_args):
    global stop_hook_id
    if command == "":
        result.AppendMessage(str(stop_hook_id))
    else:
        stop_hook_id = int(command)
