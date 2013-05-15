def foo2_function(debugger, args, result, dict):
	print >>result,  ("foo2 says " + args)
	return None

def __lldb_init_module(debugger, session_dict):
	debugger.HandleCommand("command script add -f foo2.foo2_function foo2cmd")
	return None