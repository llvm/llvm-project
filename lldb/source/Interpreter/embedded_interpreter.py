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


def generate_extension_schema(cls):
    """Introspect a scripting extension base class and return a JSON schema
    describing its members. Used by `scripting extension generate` (via
    `ScriptInterpreterPython::GetExtensionSchema`) to emit a skeleton
    subclass with `# TODO: Implement` stubs for each method the base class
    defines. Each method entry carries the signature, type hints,
    docstring, and whether it's `@abstractmethod`, so the generator can
    decide which methods to stub out (all of them with `-a`, otherwise
    just the abstract ones). The schema also lists non-callable
    attributes the base class exposes -- class-level values plus
    class-body type annotations -- so the generator can advertise them in
    the derived class' docstring. A `typing_imports` field enumerates
    every `typing` generic (`Optional`, `Union`, ...) referenced by the
    signatures or attribute types, so the generator can add the right
    `from typing import` line without having to re-scan strings."""
    import inspect, json, typing

    used_typing = set()

    def _record_typing(s):
        # Anything from `typing.__all__` referenced as a generic
        # (`Optional[...]`) gets picked up. Keying off the `[` avoids
        # matching identifiers that merely embed the name.
        if not s:
            return
        for name in typing.__all__:
            if f"{name}[" in s:
                used_typing.add(name)

    def _fmt_type(t):
        # `type(None)` stringifies as `NoneType`; render it as the
        # literal `None` so the annotation stays valid Python.
        if t is type(None):
            return "None"
        if isinstance(t, type):
            if t.__module__ == "builtins":
                return t.__name__
            return f"{t.__module__}.{t.__name__}"
        # `typing` generic aliases stringify with a leading `typing.`
        # (`typing.Optional[list]`); the module prefix is noise for a
        # docstring. `Union[int, str, None]` also renders its `None`
        # component as `NoneType`, so fix that up too.
        formatted = str(t).replace("typing.", "").replace("NoneType", "None")
        _record_typing(formatted)
        return formatted

    def _build_signature(func):
        # Reconstruct the signature from resolved type hints so forward
        # refs (`"ScriptedFrame"`) come out as their real class -- what
        # `inspect.signature(...)`'s own `str` would render as
        # `ForwardRef('ScriptedFrame')`.
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = {}
        sig = inspect.signature(func)
        parts = []
        for name, param in sig.parameters.items():
            piece = name
            if name in hints:
                piece += f": {_fmt_type(hints[name])}"
            if param.default is not inspect.Parameter.empty:
                piece += f" = {param.default!r}"
            parts.append(piece)
        rendered = "(" + ", ".join(parts) + ")"
        if "return" in hints:
            rendered += f" -> {_fmt_type(hints['return'])}"
        return rendered

    def _get_function_metadata(func):
        try:
            hints = typing.get_type_hints(func)
            type_hints = {k: str(v) for k, v in hints.items()}
        except Exception:
            type_hints = {}
        return {
            "signature": _build_signature(func),
            "type_hints": type_hints,
            "is_abstract": getattr(func, "__isabstractmethod__", False),
            "doc": inspect.getdoc(func),
        }

    try:
        class_hints = typing.get_type_hints(cls)
    except Exception:
        class_hints = {}

    members = []
    attributes = []
    seen_attrs = set()
    for name, member in inspect.getmembers(cls):
        if inspect.isfunction(member):
            members.append({"name": name, **_get_function_metadata(member)})
            continue
        if name.startswith("_"):
            continue
        entry = {"name": name}
        if name in class_hints:
            entry["type"] = _fmt_type(class_hints[name])
        attributes.append(entry)
        seen_attrs.add(name)

    # Class-body type annotations without a runtime value
    # (`target: SBTarget`) don't show up in `inspect.getmembers`, so pick
    # them up from the hint map directly.
    for name in class_hints:
        if name.startswith("_") or name in seen_attrs:
            continue
        attributes.append({"name": name, "type": _fmt_type(class_hints[name])})
        seen_attrs.add(name)

    return json.dumps(
        {
            "class": cls.__name__,
            "module": cls.__module__,
            "doc": inspect.getdoc(cls),
            "members": members,
            "attributes": attributes,
            "typing_imports": sorted(used_typing),
        },
        separators=(",", ":"),
    )


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
