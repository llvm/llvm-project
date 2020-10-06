import lldb

class stop_handler:
    def __init__(self, target, extra_args, dict):
        self.extra_args = extra_args
        self.target = target

    def handle_stop(self, exe_ctx, stream):
        stream.Print("I did indeed run\n")
        return True
