import lldb


class EchoCommand:
    def __init__(self, debugger, internal_dict):
        pass

    def __call__(self, debugger, command, exe_ctx, result):
        result.AppendMessage(command)


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        "command script add -c scripted_command.EchoCommand echo-cmd"
    )
