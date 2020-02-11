
import six


def command(debugger, command, result, internal_dict):
    result.PutCString(six.u("hello world A"))
    return None
