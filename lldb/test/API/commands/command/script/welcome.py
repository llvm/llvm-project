from __future__ import print_function
import lldb
import sys


class WelcomeCommand(object):

    def __init__(self, debugger, session_dict):
        pass

    def get_short_help(self):
        return "Just a docstring for welcome_impl\nA command that says hello to LLDB users"

    def __call__(self, debugger, args, exe_ctx, result):
        print('Hello ' + args + ', welcome to LLDB', file=result)
        return None


class TargetnameCommand(object):

    def __init__(self, debugger, session_dict):
        pass

    def __call__(self, debugger, args, exe_ctx, result):
        target = debugger.GetSelectedTarget()
        file = target.GetExecutable()
        print('Current target ' + file.GetFilename(), file=result)
        if args == 'fail':
            result.SetError('a test for error in command')

    def get_flags(self):
        return lldb.eCommandRequiresTarget


def print_wait_impl(debugger, args, result, dict):
    result.SetImmediateOutputFile(sys.stdout)
    print('Trying to do long task..', file=result)
    import time
    time.sleep(1)
    print('Still doing long task..', file=result)
    time.sleep(1)
    print('Done; if you saw the delays I am doing OK', file=result)


def check_for_synchro(debugger, args, result, dict):
    if debugger.GetAsync():
        print('I am running async', file=result)
    if debugger.GetAsync() == False:
        print('I am running sync', file=result)


def takes_exe_ctx(debugger, args, exe_ctx, result, dict):
    print(str(exe_ctx.GetTarget()), file=result)
