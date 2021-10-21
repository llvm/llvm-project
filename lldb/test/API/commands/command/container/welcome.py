from __future__ import print_function
import lldb
import sys


class WelcomeCommand(object):

    def __init__(self, debugger, session_dict):
        pass

    def get_short_help(self):
        return "Just a docstring for Welcome\nA command that says hello to LLDB users"

    def __call__(self, debugger, args, exe_ctx, result):
        print('Hello ' + args + ', welcome to LLDB', file=result)
        return None

class WelcomeCommand2(object):

    def __init__(self, debugger, session_dict):
        pass

    def get_short_help(self):
        return "Just a docstring for the second Welcome\nA command that says hello to LLDB users"

    def __call__(self, debugger, args, exe_ctx, result):
        print('Hello ' + args + ', welcome again to LLDB', file=result)
        return None
