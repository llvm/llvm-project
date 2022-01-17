import lldb

debugger_copy = None
result_copy = None

def save_debugger(debugger, command, context, result, internal_dict):
    global debugger_copy, result_copy
    debugger_copy = debugger
    result_copy = result
    result.AppendMessage(str(debugger))
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
