import sys

def welcome_impl(debugger, args, result, dict):
    """
        Just a docstring for welcome_impl
        A command that says hello to LLDB users
    """
    result.Printf('Hello ' + args + ', welcome to LLDB');
    return None;

def target_name_impl(debugger, args, result, dict):
    target = debugger.GetSelectedTarget()
    file = target.GetExecutable()
    result.PutCString('Current target ' + file.GetFilename())
    if args == 'fail':
        return 'a test for error in command'
    else:
        return None

def print_wait_impl(debugger, args, result, dict):
    result.SetImmediateOutputFile(sys.stdout)
    result.PutCString('Trying to do long task..')
    import time
    time.sleep(1)
    result.PutCString('Still doing long task..')
    time.sleep(1)
    result.PutCString('Done; if you saw the delays I am doing OK')
    return None

def check_for_synchro(debugger, args, result, dict):
    if debugger.GetAsync() == True:
        result.PutCString('I am running async')
    if debugger.GetAsync() == False:
        result.PutCString('I am running sync')
    return None
