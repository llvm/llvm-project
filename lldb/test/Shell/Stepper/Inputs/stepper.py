from __future__ import print_function
from collections import defaultdict

import os
import re

import lldb

###############################################################################
# Configurable options (see the STEPPER_OPTIONS env var)
###############################################################################

# Only visit unoptimized frames.
kOnlyUnoptimized = True

# Only visit frames with swift code.
kOnlySwift = True

# Skip doing "frame var" on all locals, in each frame.
kSkipFrameVarAllLocals = False

# Skip doing "po" on all locals, in each frame.
kSkipExprAllLocals = False

# Maximum number of times to visit a PC before "finish"ing out.
kMaxPCVisitCount = 50

# Run the process. It's useful to set this to False if you have monitor/test
# lldb's, and the child lldb can't read from stdin to do things like continue
# at a breakpoint.
kRunTheProcess = True

# Skip N frames between each inspection.
kSkipNBetweenInspections = 100

###############################################################################


def parse_options():
    """
    Parse the STEPPER_OPTIONS environment variable.

    Update any constants specified in the option string.
    """
    global kOnlyUnoptimized, kOnlySwift, kSkipFrameVarAllLocals, \
            kSkipExprAllLocals, kMaxPCVisitCount, kRunTheProcess, \
            kSkipNBetweenInspections
    opts = os.getenv('STEPPER_OPTIONS', '').split(';')
    for o in opts:
        m = re.match('(\w+)=([^;]+)', o)
        if not m:
            print('Unrecognized option:', o)
        else:
            option, val = m.groups()
            print('Setting', option, '=', val)
            globals()[option] = eval(val)


def doit(dbg, cmd):
    "Run a driver command."
    print('::', cmd)
    dbg.HandleCommand(cmd)
    return False


def doquit(dbg):
    "Quit the stepper script interpreter."
    os._exit(0)


def should_stop_stepping(process):
    "Decide whether we should stop stepping."
    state = process.GetState()
    if state in (lldb.eStateExited, lldb.eStateDetached):
        print('Process has exited or has been detached, exiting...')
        return True
    if state in (lldb.eStateCrashed, lldb.eStateInvalid):
        print('Process has crashed or is in an invalid state, exiting...')
        return True
    return False


def alter_PC(dbg, process, cmd):
    "Run a driver command that changes the PC. Return whether to stop stepping."
    # Check the process state /after/ we step. Any time we advance the PC,
    # the process state may change.
    doit(dbg, cmd)
    return should_stop_stepping(process)


def return_from_frame(thread, frame):
    print(':: Popping current frame...')
    old_name = frame.GetFunctionName()
    thread.StepOutOfFrame(frame)
    new_frame = thread.GetSelectedFrame()
    new_name = new_frame.GetFunctionName()
    print(':: Transitioned from {0} -> {1}.'.format(old_name, new_name))
    return True


def __lldb_init_module(dbg, internal_dict):
    parse_options()

    # Make each debugger command synchronous.
    dbg.SetAsync(False)

    # Run the program and stop it when it reaches main().
    if kRunTheProcess:
        if doit(dbg, 'breakpoint set -n main') or doit(dbg, 'run'):
            print(':: Failed to run the process!')
            dbg.Terminate()
            return

    # Step through the program until it exits.
    gen = 0
    target = dbg.GetSelectedTarget()
    process = target.GetProcess()
    visited_pc_counts = defaultdict(int)
    while True:
        gen += 1
        print(':: Generation', gen)

        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        do_inspection = True

        # Sometimes, lldb gets lost after stepping. This is rdar://70546777.
        # Try to remind lldb where it is by running 'frame select'.
        if not frame.GetFunctionName():
            doit(dbg, 'frame select')

        # Skip frames without valid line entries.
        line_entry = frame.GetLineEntry()
        if do_inspection and not line_entry.IsValid():
            do_inspection = False

        # Skip optimized frames if asked to do so.
        if do_inspection and kOnlyUnoptimized and str(frame).endswith(
                ' [opt]'):
            do_inspection = False

        # Skip non-Swift frames if asked to do so.
        skip_inspection_due_to_frame_lang = False
        if do_inspection and kOnlySwift and \
                frame.GuessLanguage() != lldb.eLanguageTypeSwift:
            do_inspection = False
            skip_inspection_due_to_frame_lang = True

        if kSkipNBetweenInspections > 0 and gen % kSkipNBetweenInspections != 0:
            do_inspection = False

        # Don't inspect the same PC twice. Some version of this is needed to
        # make speedy progress on programs containing loops or recursion. The
        # tradeoff is that we lose test coverage (the objects visible at this
        # PC may change over time).
        cur_pc = frame.GetPC()
        visit_count = visited_pc_counts[cur_pc]
        visited_pc_counts[cur_pc] += 1
        if do_inspection and visit_count > 0:
            do_inspection = False

        # Inspect the current frame if permitted to do so.
        if do_inspection:
            doit(dbg, 'bt')

            # Exercise `frame variable`.
            if not kSkipFrameVarAllLocals:
                doit(dbg, 'frame variable')

            # Exercise `expr -O`.
            if not kSkipExprAllLocals:
                get_args = True
                get_locals = True
                get_statics = True
                get_in_scope_only = True
                variables = frame.GetVariables(get_args, get_locals,
                                               get_statics, get_in_scope_only)
                for var in variables:
                    name = var.GetName()
                    if not var.GetLocation():
                        continue
                    doit(dbg, 'po {0}'.format(name))

        # Sometimes, we might visit a PC way too often (or there's nothing for
        # us to inspect because there's no line entry). After the first visit of
        # a PC, we aren't even inspecting the frame.
        #
        # To speed things up, we "finish" out of the frame if we think we've
        # spent too much time under it. The tradeoff is that we lose test
        # coverage (we may fail to step through certain program paths). That's
        # probably ok, considering that this can help *increase* test coverage
        # by virtue of helping us not get stuck in a hot loop sinkhole.
        if visit_count >= kMaxPCVisitCount or \
                skip_inspection_due_to_frame_lang or not line_entry.IsValid():
            old_func_name = frame.GetFunctionName()
            if not old_func_name:
                print(':: Stepped to frame without function name!!!')
                doquit(dbg)
                return

            while frame.GetFunctionName() == old_func_name:
                if not return_from_frame(thread, frame):
                    print(':: Failed to step out of frame!')
                    doquit(dbg)
                    return
                doit(dbg, 'frame select')
                frame = thread.GetSelectedFrame()
            continue

        if alter_PC(dbg, process, 'step'):
            print(':: Failed to step!')
            doquit(dbg)
            return
