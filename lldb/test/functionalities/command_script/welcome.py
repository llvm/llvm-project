import sys

def welcome_impl(debugger, args, result, dict):
    """
        Just a docstring for welcome_impl
        A command that says hello to LLDB users
    """
    print >>result,  ('Hello ' + args + ', welcome to LLDB');
    return None;

def target_name_impl(debugger, args, result, dict):
    target = debugger.GetSelectedTarget()
    file = target.GetExecutable()
    print >>result,  ('Current target ' + file.GetFilename())
    if args == 'fail':
        return 'a test for error in command'
    else:
        return None

def print_wait_impl(debugger, args, result, dict):
    result.SetImmediateOutputFile(sys.stdout)
    print >>result,  ('Trying to do long task..')
    import time
    time.sleep(1)
    print >>result,  ('Still doing long task..')
    time.sleep(1)
    print >>result,  ('Done; if you saw the delays I am doing OK')
    return None

def check_for_synchro(debugger, args, result, dict):
    if debugger.GetAsync() == True:
        print >>result,  ('I am running async')
    if debugger.GetAsync() == False:
        print >>result,  ('I am running sync')
    return None
