"""
Test that SBFrame::GetVariables() calls work correctly.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil


def get_names_from_value_list(value_list):
    names = list()
    for value in value_list:
        names.append(value.GetName())
    return names


class TestGetVariables(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.source = 'main.c'

    def verify_variable_names(self, description, value_list, names):
        copy_names = list(names)
        actual_names = get_names_from_value_list(value_list)
        for name in actual_names:
            if name in copy_names:
                copy_names.remove(name)
            else:
                self.assertTrue(
                    False, "didn't find '%s' in %s" %
                    (name, copy_names))
        self.assertEqual(
            len(copy_names), 0, "%s: we didn't find variables: %s in value list (%s)" %
            (description, copy_names, actual_names))

    def test(self):
        self.build()

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        line1 = line_number(self.source, '// breakpoint 1')
        line2 = line_number(self.source, '// breakpoint 2')
        line3 = line_number(self.source, '// breakpoint 3')

        breakpoint1 = target.BreakpointCreateByLocation(self.source, line1)
        breakpoint2 = target.BreakpointCreateByLocation(self.source, line2)
        breakpoint3 = target.BreakpointCreateByLocation(self.source, line3)

        self.assertTrue(breakpoint1.GetNumLocations() >= 1, PROCESS_IS_VALID)
        self.assertTrue(breakpoint2.GetNumLocations() >= 1, PROCESS_IS_VALID)
        self.assertTrue(breakpoint3.GetNumLocations() >= 1, PROCESS_IS_VALID)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        arguments = None
        environment = None

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            arguments, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint1)
        self.assertEqual(
            len(threads),
            1,
            "There should be a thread stopped at breakpoint 1")

        thread = threads[0]
        self.assertTrue(thread.IsValid(), "Thread must be valid")
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame must be valid")

        arg_names = ['argc', 'argv']
        local_names = ['i', 'j', 'k']
        static_names = ['static_var', 'g_global_var', 'g_static_var']
        breakpoint1_locals = ['i']
        breakpoint1_statics = ['static_var']
        num_args = len(arg_names)
        num_locals = len(local_names)
        num_statics = len(static_names)
        args_yes = True
        args_no = False
        locals_yes = True
        locals_no = False
        statics_yes = True
        statics_no = False
        in_scopy_only = True
        ignore_scope = False

        # Verify if we ask for only arguments that we got what we expect
        vars = frame.GetVariables(
            args_yes, locals_no, statics_no, ignore_scope)
        self.assertEqual(
            vars.GetSize(),
            num_args,
            "There should be %i arguments, but we are reporting %i" %
            (num_args,
             vars.GetSize()))
        self.verify_variable_names("check names of arguments", vars, arg_names)
        self.assertEqual(
            len(arg_names),
            num_args,
            "make sure verify_variable_names() didn't mutate list")

        # Verify if we ask for only locals that we got what we expect
        vars = frame.GetVariables(
            args_no, locals_yes, statics_no, ignore_scope)
        self.assertEqual(
            vars.GetSize(),
            num_locals,
            "There should be %i local variables, but we are reporting %i" %
            (num_locals,
             vars.GetSize()))
        self.verify_variable_names("check names of locals", vars, local_names)

        # Verify if we ask for only statics that we got what we expect
        vars = frame.GetVariables(
            args_no, locals_no, statics_yes, ignore_scope)
        print('statics: ', str(vars))
        self.assertEqual(
            vars.GetSize(),
            num_statics,
            "There should be %i static variables, but we are reporting %i" %
            (num_statics,
             vars.GetSize()))
        self.verify_variable_names(
            "check names of statics", vars, static_names)

        # Verify if we ask for arguments and locals that we got what we expect
        vars = frame.GetVariables(
            args_yes, locals_yes, statics_no, ignore_scope)
        desc = 'arguments + locals'
        names = arg_names + local_names
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)

        # Verify if we ask for arguments and statics that we got what we expect
        vars = frame.GetVariables(
            args_yes, locals_no, statics_yes, ignore_scope)
        desc = 'arguments + statics'
        names = arg_names + static_names
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)

        # Verify if we ask for locals and statics that we got what we expect
        vars = frame.GetVariables(
            args_no, locals_yes, statics_yes, ignore_scope)
        desc = 'locals + statics'
        names = local_names + static_names
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)

        # Verify if we ask for arguments, locals and statics that we got what
        # we expect
        vars = frame.GetVariables(
            args_yes, locals_yes, statics_yes, ignore_scope)
        desc = 'arguments + locals + statics'
        names = arg_names + local_names + static_names
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)

        # Verify if we ask for in scope locals that we got what we expect
        vars = frame.GetVariables(
            args_no, locals_yes, statics_no, in_scopy_only)
        desc = 'in scope locals at breakpoint 1'
        names = ['i']
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)

        # Continue to breakpoint 2
        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint2)
        self.assertEqual(
            len(threads),
            1,
            "There should be a thread stopped at breakpoint 2")

        thread = threads[0]
        self.assertTrue(thread.IsValid(), "Thread must be valid")
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame must be valid")

        # Verify if we ask for in scope locals that we got what we expect
        vars = frame.GetVariables(
            args_no, locals_yes, statics_no, in_scopy_only)
        desc = 'in scope locals at breakpoint 2'
        names = ['i', 'j']
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)

        # Continue to breakpoint 3
        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint3)
        self.assertEqual(
            len(threads),
            1,
            "There should be a thread stopped at breakpoint 3")

        thread = threads[0]
        self.assertTrue(thread.IsValid(), "Thread must be valid")
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame must be valid")

        # Verify if we ask for in scope locals that we got what we expect
        vars = frame.GetVariables(
            args_no, locals_yes, statics_no, in_scopy_only)
        desc = 'in scope locals at breakpoint 3'
        names = ['i', 'j', 'k']
        count = len(names)
        self.assertEqual(
            vars.GetSize(),
            count,
            "There should be %i %s (%s) but we are reporting %i (%s)" %
            (count,
             desc,
             names,
             vars.GetSize(),
             get_names_from_value_list(vars)))
        self.verify_variable_names("check names of %s" % (desc), vars, names)
