import lldb

# bunch of different kinds of python callables that should
# all work as commands.


def check(debugger, command, context, result, internal_dict):
    if (
        not isinstance(debugger, lldb.SBDebugger)
        or not isinstance(command, str)
        or not isinstance(result, lldb.SBCommandReturnObject)
        or not isinstance(internal_dict, dict)
        or (not context is None and not isinstance(context, lldb.SBExecutionContext))
    ):
        raise Exception()
    result.AppendMessage("All good.")


def vfoobar(*args):
    check(*args)


def v5foobar(debugger, command, context, result, internal_dict, *args):
    check(debugger, command, context, result, internal_dict)


def foobar(debugger, command, context, result, internal_dict):
    check(debugger, command, context, result, internal_dict)


def foobar4(debugger, command, result, internal_dict):
    check(debugger, command, None, result, internal_dict)


class FooBar:
    @staticmethod
    def sfoobar(debugger, command, context, result, internal_dict):
        check(debugger, command, context, result, internal_dict)

    @classmethod
    def cfoobar(cls, debugger, command, context, result, internal_dict):
        check(debugger, command, context, result, internal_dict)

    def ifoobar(self, debugger, command, context, result, internal_dict):
        check(debugger, command, context, result, internal_dict)

    def __call__(self, debugger, command, context, result, internal_dict):
        check(debugger, command, context, result, internal_dict)

    @staticmethod
    def sfoobar4(debugger, command, result, internal_dict):
        check(debugger, command, None, result, internal_dict)

    @classmethod
    def cfoobar4(cls, debugger, command, result, internal_dict):
        check(debugger, command, None, result, internal_dict)

    def ifoobar4(self, debugger, command, result, internal_dict):
        check(debugger, command, None, result, internal_dict)


class FooBar4:
    def __call__(self, debugger, command, result, internal_dict):
        check(debugger, command, None, result, internal_dict)


FooBarObj = FooBar()

FooBar4Obj = FooBar4()
