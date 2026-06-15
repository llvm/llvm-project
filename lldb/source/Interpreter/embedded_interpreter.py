import sys
import builtins
import code
import lldb
import traceback

try:
    import readline
    import rlcompleter
except ImportError:
    have_readline = False
except AttributeError:
    # This exception gets hit by the rlcompleter when Linux is using
    # the readline suppression import.
    have_readline = False
else:
    have_readline = True

    def is_libedit():
        if hasattr(readline, "backend"):
            return readline.backend == "editline"
        return "libedit" in getattr(readline, "__doc__", "")

    if is_libedit():
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

# When running one line, we might place the string to run in this string
# in case it would be hard to correctly escape a string's contents

g_run_one_line_str = None


class LLDBExit(SystemExit):
    pass


def strip_and_check_exit(line):
    line = line.rstrip()
    if line in ("exit", "quit"):
        raise LLDBExit
    return line


def readfunc(prompt):
    line = input(prompt)
    return strip_and_check_exit(line)


def readfunc_stdio(prompt):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    # Readline always includes a trailing newline character unless the file
    # ends with an incomplete line. An empty line indicates EOF.
    if not line:
        raise EOFError
    return strip_and_check_exit(line)


def run_python_interpreter(local_dict):
    # Pass in the dictionary, for continuity from one session to the next.
    try:
        banner = "Python Interactive Interpreter. To exit, type 'quit()', 'exit()'."
        input_func = readfunc_stdio

        is_atty = sys.stdin.isatty()
        if is_atty:
            banner = "Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D."
            input_func = readfunc

        code.interact(banner=banner, readfunc=input_func, local=local_dict)
    except LLDBExit:
        pass
    except SystemExit as e:
        if e.code:
            print("Script exited with code %s" % e.code)

def run_one_line(local_dict, input_string):
    global g_run_one_line_str
    try:
        input_string = strip_and_check_exit(input_string)
        repl = code.InteractiveConsole(local_dict)
        if input_string:
            # A newline is appended to support one-line statements containing
            # control flow. For example "if True: print(1)" silently does
            # nothing, but works with a newline: "if True: print(1)\n".
            input_string += "\n"
            repl.runsource(input_string)
        elif g_run_one_line_str:
            repl.runsource(g_run_one_line_str)
    except LLDBExit:
        pass
    except SystemExit as e:
        if e.code:
            print("Script exited with code %s" % e.code)
