import lldb


class EchoCommand:
    def __init__(self, debugger, internal_dict):
        pass

    def __call__(self, debugger, command, exe_ctx, result):
        result.AppendMessage(command)
