import lldb


def some_command_here(debugger, command, result, d):
    if command == "a":
        print("Script command invoked", file=result)
        return True
    else:
        print("Script command received unexpected arguments", file=result)
        return False
