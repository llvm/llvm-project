import lldb


@lldb.command()
def throw(debugger, cmd, ctx, result, _):
    raise Exception("command failed")
