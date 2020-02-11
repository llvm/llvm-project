def bug11569(debugger, args, result, dict):
    """
    http://llvm.org/bugs/show_bug.cgi?id=11569
    LLDBSwigPythonCallCommand crashes when a command script returns an object.
    """
    return ["return", "a", "non-string", "should", "not", "crash", "LLDB"]
