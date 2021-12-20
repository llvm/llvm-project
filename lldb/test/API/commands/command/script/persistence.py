import lldb

debugger_copy = None

def save_debugger(debugger, command, context, result, internal_dict):
    global debugger_copy
    debugger_copy = debugger
    result.AppendMessage(str(debugger))
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
