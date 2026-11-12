#!/usr/bin/env python

# ---------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# # To use this in the embedded python interpreter using "lldb" just
# import it with the full path using the "command script import"
# command
#   (lldb) command script import /path/to/cmdtemplate.py
# ---------------------------------------------------------------------

import inspect
import lldb
import argparse
import shlex
import sys

# Each new breakpoint gets a unique ID starting from 1.
nextid = 1
# List of breakpoint set from python, the key is the ID and the value the
# actual breakpoint. These are NOT LLDB SBBreakpoint objects.
breakpoints = dict()

exprOptions = lldb.SBExpressionOptions()
exprOptions.SetIgnoreBreakpoints()
exprOptions.SetLanguage(lldb.eLanguageTypeC)


class MlirDebug:
    """MLIR debugger commands
    This is the class that hooks into LLDB and registers the `mlir` command.
    Other providers can register subcommands below this one.
    """

    lldb_command = "mlir"
    parser = None

    def __init__(self, debugger, unused):
        super().__init__()
        self.create_options()
        self.help_string = MlirDebug.parser.format_help()

    @classmethod
    def create_options(cls):
        if MlirDebug.parser:
            return MlirDebug.parser
        usage = "usage: %s [options]" % (cls.lldb_command)
        description = "TODO."

        # Pass add_help_option = False, since this keeps the command in line
        # with lldb commands, and we wire up "help command" to work by
        # providing the long & short help methods below.
        MlirDebug.parser = argparse.ArgumentParser(
            prog=cls.lldb_command, usage=usage, description=description, add_help=False
        )
        MlirDebug.subparsers = MlirDebug.parser.add_subparsers(dest="command")
        return MlirDebug.parser

    def get_short_help(self):
        return "MLIR debugger commands"

    def get_long_help(self):
        return self.help_string

    def __call__(self, debugger, command, exe_ctx, result):
        # Use the Shell Lexer to properly parse up command options just like a
        # shell would
        command_args = shlex.split(command)

        try:
            args = MlirDebug.parser.parse_args(command_args)
        except:
            result.SetError("option parsing failed")
            raise
        args.func(args, debugger, command, exe_ctx, result)

    @classmethod
    def on_process_start(frame, bp_loc, dict):
        print("Process started")


class SetControl:
    # Define the subcommands that controls what to do when a breakpoint is hit.
    # The key is the subcommand name, the value is a tuple of the command ID to
    # pass to MLIR and the help string.
    commands = {
        "apply": (1, "Apply the current action and continue the execution"),
        "skip": (2, "Skip the current action and continue the execution"),
        "step": (3, "Step into the current action"),
        "next": (4, "Step over the current action"),
        "finish": (5, "Step out of the current action"),
    }

    @classmethod
    def register_mlir_subparser(cls):
        for cmd, (cmdInt, help) in cls.commands.items():
            parser = MlirDebug.subparsers.add_parser(
                cmd,
                help=help,
            )
            parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError("No valid frame (program not running?)")
            return
        cmdInt = cls.commands.get(options.command, None)
        if not cmdInt:
            result.SetError("Invalid command: %s" % (options.command))
            return

        result = frame.EvaluateExpression(
            "((bool (*)(int))mlirDebuggerSetControl)(%d)" % (cmdInt[0]),
            exprOptions,
        )
        if not result.error.Success():
            print("Error setting up command: %s" % (result.error))
            return
        debugger.SetAsync(True)
        result = exe_ctx.GetProcess().Continue()
        debugger.SetAsync(False)


class PrintContext:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "context", help="Print the current context"
        )
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError("Can't print context without a valid frame")
            return
        result = frame.EvaluateExpression(
            "((bool (*)())&mlirDebuggerPrintContext)()", exprOptions
        )
        if not result.error.Success():
            print("Error printing context: %s" % (result.error))
            return


class Backtrace:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "backtrace", aliases=["bt"], help="Print the current backtrace"
        )
        cls.parser.set_defaults(func=cls.process_options)
        cls.parser.add_argument("--context", default=False, action="store_true")

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't backtrace without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)(bool))mlirDebuggerPrintActionBacktrace)(%d)" % (options.context),
            exprOptions,
        )
        if not result.error.Success():
            print("Error printing breakpoints: %s" % (result.error))
            return


###############################################################################
# Cursor manipulation
###############################################################################


class PrintCursor:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "cursor-print", aliases=["cursor-p"], help="Print the current cursor"
        )
        cls.parser.add_argument(
            "--print-region", "--regions", "-r", default=False, action="store_true"
        )
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't print cursor without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)(bool))mlirDebuggerCursorPrint)(%d)" % (options.print_region),
            exprOptions,
        )
        if not result.error.Success():
            print("Error printing cursor: %s" % (result.error))
            return


class SelectCursorFromContext:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "cursor-select-from-context",
            aliases=["cursor-s"],
            help="Select the cursor from the current context",
        )
        cls.parser.add_argument("index", type=int, help="Index in the context")
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't manipulate cursor without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)(int))mlirDebuggerCursorSelectIRUnitFromContext)(%d)"
            % options.index,
            exprOptions,
        )
        if not result.error.Success():
            print("Error manipulating cursor: %s" % (result.error))
            return


class CursorSelectParent:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "cursor-parent", aliases=["cursor-up"], help="Select the cursor parent"
        )
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't manipulate cursor without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)())mlirDebuggerCursorSelectParentIRUnit)()",
            exprOptions,
        )
        if not result.error.Success():
            print("Error manipulating cursor: %s" % (result.error))
            return


class SelectCursorChild:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "cursor-child", aliases=["cursor-c"], help="Select the nth child"
        )
        cls.parser.add_argument("index", type=int, help="Index of the child to select")
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't manipulate cursor without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)(int))mlirDebuggerCursorSelectChildIRUnit)(%d)" % options.index,
            exprOptions,
        )
        if not result.error.Success():
            print("Error manipulating cursor: %s" % (result.error))
            return


class CursorSelecPrevious:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "cursor-previous",
            aliases=["cursor-prev"],
            help="Select the cursor previous element",
        )
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't manipulate cursor without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)())mlirDebuggerCursorSelectPreviousIRUnit)()",
            exprOptions,
        )
        if not result.error.Success():
            print("Error manipulating cursor: %s" % (result.error))
            return


class CursorSelecNext:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "cursor-next", aliases=["cursor-n"], help="Select the cursor next element"
        )
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        frame = exe_ctx.GetFrame()
        if not frame.IsValid():
            result.SetError(
                "Can't manipulate cursor without a valid frame (program not running?)"
            )
        result = frame.EvaluateExpression(
            "((bool(*)())mlirDebuggerCursorSelectNextIRUnit)()",
            exprOptions,
        )
        if not result.error.Success():
            print("Error manipulating cursor: %s" % (result.error))
            return


###############################################################################
# Breakpoints
###############################################################################


class EnableBreakpoint:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "enable", help="Enable a single breakpoint (given its ID)"
        )
        cls.parser.add_argument("id", help="ID of the breakpoint to enable")
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        bp = breakpoints.get(int(options.id), None)
        if not bp:
            result.SetError("No breakpoint with ID %d" % int(options.id))
            return
        bp.enable(exe_ctx.GetFrame())


class DisableBreakpoint:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "disable", help="Disable a single breakpoint (given its ID)"
        )
        cls.parser.add_argument("id", help="ID of the breakpoint to disable")
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        bp = breakpoints.get(int(options.id), None)
        if not bp:
            result.SetError("No breakpoint with ID %s" % options.id)
            return
        bp.disable(exe_ctx.GetFrame())


class ListBreakpoints:
    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            "list", help="List all current breakpoints"
        )
        cls.parser.set_defaults(func=cls.process_options)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        for id, bp in sorted(breakpoints.items()):
            print(id, type(id), str(bp), "enabled" if bp.isEnabled else "disabled")


class Breakpoint:
    def __init__(self):
        global nextid
        self.id = nextid
        nextid += 1
        breakpoints[self.id] = self
        self.isEnabled = True

    def enable(self, frame=None):
        self.isEnabled = True
        if not frame or not frame.IsValid():
            return
        # use a C cast to force the type of the breakpoint handle to be void * so
        # that we don't rely on DWARF. Also add a fake bool return value otherwise
        # LLDB can't signal any error with the expression evaluation (at least I don't know how).
        cmd = (
            "((bool (*)(void *))mlirDebuggerEnableBreakpoint)((void *)%s)" % self.handle
        )
        result = frame.EvaluateExpression(cmd, exprOptions)
        if not result.error.Success():
            print("Error enabling breakpoint: %s" % (result.error))
            return

    def disable(self, frame=None):
        self.isEnabled = False
        if not frame or not frame.IsValid():
            return
        # use a C cast to force the type of the breakpoint handle to be void * so
        # that we don't rely on DWARF. Also add a fake bool return value otherwise
        # LLDB can't signal any error with the expression evaluation (at least I don't know how).
        cmd = (
            "((bool (*)(void *)) mlirDebuggerDisableBreakpoint)((void *)%s)"
            % self.handle
        )
        result = frame.EvaluateExpression(cmd, exprOptions)
        if not result.error.Success():
            print("Error disabling breakpoint: %s" % (result.error))
            return


class TagBreakpoint(Breakpoint):
    mlir_subcommand = "break-on-tag"

    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def __str__(self):
        return "[%d] TagBreakpoint(%s)" % (self.id, self.tag)

    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            cls.mlir_subcommand, help="add a breakpoint on actions' tag matching"
        )
        cls.parser.set_defaults(func=cls.process_options)
        cls.parser.add_argument("tag", help="tag to match")

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        breakpoint = TagBreakpoint(options.tag)
        print("Added breakpoint %s" % str(breakpoint))

        frame = exe_ctx.GetFrame()
        if frame.IsValid():
            breakpoint.install(frame)

    def install(self, frame):
        result = frame.EvaluateExpression(
            '((void *(*)(const char *))mlirDebuggerAddTagBreakpoint)("%s")'
            % (self.tag),
            exprOptions,
        )
        if not result.error.Success():
            print("Error installing breakpoint: %s" % (result.error))
            return
        # Save the handle, this is necessary to implement enable/disable.
        self.handle = result.GetValue()


class FileLineBreakpoint(Breakpoint):
    mlir_subcommand = "break-on-file"

    def __init__(self, file, line, col):
        super().__init__()
        self.file = file
        self.line = line
        self.col = col

    def __str__(self):
        return "[%d] FileLineBreakpoint(%s, %d, %d)" % (
            self.id,
            self.file,
            self.line,
            self.col,
        )

    @classmethod
    def register_mlir_subparser(cls):
        cls.parser = MlirDebug.subparsers.add_parser(
            cls.mlir_subcommand,
            help="add a breakpoint that filters on location of the IR affected by an action. The syntax is file:line:col where file and col are optional",
        )
        cls.parser.set_defaults(func=cls.process_options)
        cls.parser.add_argument("location", type=str)

    @classmethod
    def process_options(cls, options, debugger, command, exe_ctx, result):
        split_loc = options.location.split(":")
        file = split_loc[0]
        line = int(split_loc[1]) if len(split_loc) > 1 else -1
        col = int(split_loc[2]) if len(split_loc) > 2 else -1
        breakpoint = FileLineBreakpoint(file, line, col)
        print("Added breakpoint %s" % str(breakpoint))

        frame = exe_ctx.GetFrame()
        if frame.IsValid():
            breakpoint.install(frame)

    def install(self, frame):
        result = frame.EvaluateExpression(
            '((void *(*)(const char *, int, int))mlirDebuggerAddFileLineColLocBreakpoint)("%s", %d, %d)'
            % (self.file, self.line, self.col),
            exprOptions,
        )
        if not result.error.Success():
            print("Error installing breakpoint: %s" % (result.error))
            return
        # Save the handle, this is necessary to implement enable/disable.
        self.handle = result.GetValue()


def on_start(frame, bpno, err):
    print("MLIR debugger attaching...")
    for _, bp in sorted(breakpoints.items()):
        if bp.isEnabled:
            print("Installing breakpoint %s" % (str(bp)))
            bp.install(frame)
        else:
            print("Skipping disabled breakpoint %s" % (str(bp)))

    return True


def __lldb_init_module(debugger, dict):
    target = debugger.GetTargetAtIndex(0)
    debugger.SetAsync(False)
    if not target:
        print("No target is loaded, please load a target before loading this script.")
        return
    if debugger.GetNumTargets() > 1:
        print(
            "Multiple targets (%s) loaded, attaching MLIR debugging to %s"
            % (debugger.GetNumTargets(), target)
        )

    # Register all classes that have a register_lldb_command method
    module_name = __name__
    parser = MlirDebug.create_options()
    MlirDebug.__doc__ = parser.format_help()

    # Add the MLIR entry point to LLDB as a command.
    command = "command script add -o -c %s.%s %s" % (
        module_name,
        MlirDebug.__name__,
        MlirDebug.lldb_command,
    )
    debugger.HandleCommand(command)

    main_bp = target.BreakpointCreateByName("main")
    main_bp.SetScriptCallbackFunction("action_debugging.on_start")
    main_bp.SetAutoContinue(auto_continue=True)

    on_breackpoint = target.BreakpointCreateByName("mlirDebuggerBreakpointHook")

    print(
        'The "{0}" command has been installed for target `{1}`, type "help {0}" or "{0} '
        '--help" for detailed help.'.format(MlirDebug.lldb_command, target)
    )
    for _name, cls in inspect.getmembers(sys.modules[module_name]):
        if inspect.isclass(cls) and getattr(cls, "register_mlir_subparser", None):
            cls.register_mlir_subparser()
