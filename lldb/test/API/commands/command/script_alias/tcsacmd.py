import lldb


def some_command_here(debugger, command, result, d):
    if command == "a":
        print("Victory is mine", file=result)
        return True
    else:
        print("Sadness for all", file=result)
        return False
