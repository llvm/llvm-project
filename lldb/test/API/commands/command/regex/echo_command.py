import lldb

def echo_command(debugger, args, result, dict):
    result.Print(args+'\n')
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
    return True
