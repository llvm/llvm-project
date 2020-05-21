"""
Test SBTarget APIs.
"""

from __future__ import print_function


import unittest2
import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TargetAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number(
            'main.c', '// Find the line number for breakpoint 1 here.')
        self.line2 = line_number(
            'main.c', '// Find the line number for breakpoint 2 here.')
        self.line_main = line_number(
            "main.c", "// Set a break at entry to main.")

    # rdar://problem/9700873
    # Find global variable value fails for dwarf if inferior not started
    # (Was CrashTracer: [USER] 1 crash in Python at _lldb.so: lldb_private::MemoryCache::Read + 94)
    #
    # It does not segfaults now.  But for dwarf, the variable value is None if
    # the inferior process does not exist yet.  The radar has been updated.
    #@unittest232.skip("segmentation fault -- skipping")
    @add_test_categories(['pyapi'])
    def test_find_global_variables(self):
        """Exercise SBTarget.FindGlobalVariables() API."""
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_global_variables('b.out')

    @add_test_categories(['pyapi'])
    def test_find_compile_units(self):
        """Exercise SBTarget.FindCompileUnits() API."""
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_compile_units(self.getBuildArtifact('b.out'))

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_find_functions(self):
        """Exercise SBTarget.FindFunctions() API."""
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_functions('b.out')

    @add_test_categories(['pyapi'])
    def test_get_description(self):
        """Exercise SBTarget.GetDescription() API."""
        self.build()
        self.get_description()

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber='llvm.org/pr21765')
    def test_resolve_symbol_context_with_address(self):
        """Exercise SBTarget.ResolveSymbolContextForAddress() API."""
        self.build()
        self.resolve_symbol_context_with_address()

    @add_test_categories(['pyapi'])
    def test_get_platform(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('b.out')
        platform = target.platform
        self.assertTrue(platform, VALID_PLATFORM)

    @add_test_categories(['pyapi'])
    def test_get_data_byte_size(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('b.out')
        self.assertEqual(target.data_byte_size, 1)

    @add_test_categories(['pyapi'])
    def test_get_code_byte_size(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('b.out')
        self.assertEqual(target.code_byte_size, 1)

    @add_test_categories(['pyapi'])
    def test_resolve_file_address(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('b.out')

        # find the file address in the .data section of the main
        # module
        data_section = self.find_data_section(target)
        data_section_addr = data_section.file_addr

        # resolve the above address, and compare the address produced
        # by the resolution against the original address/section
        res_file_addr = target.ResolveFileAddress(data_section_addr)
        self.assertTrue(res_file_addr.IsValid())

        self.assertEqual(data_section_addr, res_file_addr.file_addr)

        data_section2 = res_file_addr.section
        self.assertIsNotNone(data_section2)
        self.assertEqual(data_section.name, data_section2.name)

    @add_test_categories(['pyapi'])
    @skipIfReproducer # SBTarget::ReadMemory is not instrumented.
    def test_read_memory(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('b.out')

        breakpoint = target.BreakpointCreateByLocation(
            "main.c", self.line_main)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Put debugger into synchronous mode so when we target.LaunchSimple returns
        # it will guaranteed to be at the breakpoint
        self.dbg.SetAsync(False)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        # find the file address in the .data section of the main
        # module
        data_section = self.find_data_section(target)
        sb_addr = lldb.SBAddress(data_section, 0)
        error = lldb.SBError()
        content = target.ReadMemory(sb_addr, 1, error)
        self.assertTrue(error.Success(), "Make sure memory read succeeded")
        self.assertEqual(len(content), 1)

    def create_simple_target(self, fn):
        exe = self.getBuildArtifact(fn)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        return target

    def find_data_section(self, target):
        mod = target.GetModuleAtIndex(0)
        data_section = None
        for s in mod.sections:
            sect_type = s.GetSectionType()
            if sect_type == lldb.eSectionTypeData:
                data_section = s
                break
            elif sect_type == lldb.eSectionTypeContainer:
                for i in range(s.GetNumSubSections()):
                    ss = s.GetSubSectionAtIndex(i)
                    sect_type = ss.GetSectionType()
                    if sect_type == lldb.eSectionTypeData:
                        data_section = ss
                        break

        self.assertIsNotNone(data_section)
        return data_section

    def find_global_variables(self, exe_name):
        """Exercise SBTaget.FindGlobalVariables() API."""
        exe = self.getBuildArtifact(exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # rdar://problem/9700873
        # Find global variable value fails for dwarf if inferior not started
        # (Was CrashTracer: [USER] 1 crash in Python at _lldb.so: lldb_private::MemoryCache::Read + 94)
        #
        # Remove the lines to create a breakpoint and to start the inferior
        # which are workarounds for the dwarf case.

        breakpoint = target.BreakpointCreateByLocation('main.c', self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        # Make sure we hit our breakpoint:
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)
        self.assertTrue(len(thread_list) == 1)

        value_list = target.FindGlobalVariables(
            'my_global_var_of_char_type', 3)
        self.assertTrue(value_list.GetSize() == 1)
        my_global_var = value_list.GetValueAtIndex(0)
        self.DebugSBValue(my_global_var)
        self.assertTrue(my_global_var)
        self.expect(my_global_var.GetName(), exe=False,
                    startstr="my_global_var_of_char_type")
        self.expect(my_global_var.GetTypeName(), exe=False,
                    startstr="char")
        self.expect(my_global_var.GetValue(), exe=False,
                    startstr="'X'")


        if not configuration.is_reproducer():
            # While we are at it, let's also exercise the similar
            # SBModule.FindGlobalVariables() API.
            for m in target.module_iter():
                if os.path.normpath(m.GetFileSpec().GetDirectory()) == self.getBuildDir() and m.GetFileSpec().GetFilename() == exe_name:
                    value_list = m.FindGlobalVariables(
                        target, 'my_global_var_of_char_type', 3)
                    self.assertTrue(value_list.GetSize() == 1)
                    self.assertTrue(
                        value_list.GetValueAtIndex(0).GetValue() == "'X'")
                    break

    def find_compile_units(self, exe):
        """Exercise SBTarget.FindCompileUnits() API."""
        source_name = "main.c"

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        list = target.FindCompileUnits(lldb.SBFileSpec(source_name, False))
        # Executable has been built just from one source file 'main.c',
        # so we may check only the first element of list.
        self.assertTrue(
            list[0].GetCompileUnit().GetFileSpec().GetFilename() == source_name)

    def find_functions(self, exe_name):
        """Exercise SBTaget.FindFunctions() API."""
        exe = self.getBuildArtifact(exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        list = target.FindFunctions('c', lldb.eFunctionNameTypeAuto)
        self.assertTrue(list.GetSize() == 1)

        for sc in list:
            self.assertTrue(
                sc.GetModule().GetFileSpec().GetFilename() == exe_name)
            self.assertTrue(sc.GetSymbol().GetName() == 'c')

    def get_description(self):
        """Exercise SBTaget.GetDescription() API."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        from lldbsuite.test.lldbutil import get_description

        # get_description() allows no option to mean
        # lldb.eDescriptionLevelBrief.
        desc = get_description(target)
        #desc = get_description(target, option=lldb.eDescriptionLevelBrief)
        if not desc:
            self.fail("SBTarget.GetDescription() failed")
        self.expect(desc, exe=False,
                    substrs=['a.out'])
        self.expect(desc, exe=False, matching=False,
                    substrs=['Target', 'Module', 'Breakpoint'])

        desc = get_description(target, option=lldb.eDescriptionLevelFull)
        if not desc:
            self.fail("SBTarget.GetDescription() failed")
        self.expect(desc, exe=False,
                    substrs=['a.out', 'Target', 'Module', 'Breakpoint'])

    @not_remote_testsuite_ready
    @add_test_categories(['pyapi'])
    @no_debug_info_test
    @skipIfReproducer # Inferior doesn't run during replay.
    def test_launch_new_process_and_redirect_stdout(self):
        """Exercise SBTaget.Launch() API with redirected stdout."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Add an extra twist of stopping the inferior in a breakpoint, and then continue till it's done.
        # We should still see the entire stdout redirected once the process is
        # finished.
        line = line_number('main.c', '// a(3) -> c(3)')
        breakpoint = target.BreakpointCreateByLocation('main.c', line)

        # Now launch the process, do not stop at entry point, and redirect stdout to "stdout.txt" file.
        # The inferior should run to completion after "process.Continue()"
        # call.
        local_path = self.getBuildArtifact("stdout.txt")
        if os.path.exists(local_path):
            os.remove(local_path)

        if lldb.remote_platform:
            stdout_path = lldbutil.append_to_process_working_directory(self,
                "lldb-stdout-redirect.txt")
        else:
            stdout_path = local_path
        error = lldb.SBError()
        process = target.Launch(
            self.dbg.GetListener(),
            None,
            None,
            None,
            stdout_path,
            None,
            None,
            0,
            False,
            error)
        process.Continue()
        #self.runCmd("process status")
        if lldb.remote_platform:
            # copy output file to host
            lldb.remote_platform.Get(
                lldb.SBFileSpec(stdout_path),
                lldb.SBFileSpec(local_path))

        # The 'stdout.txt' file should now exist.
        self.assertTrue(
            os.path.isfile(local_path),
            "'stdout.txt' exists due to redirected stdout via SBTarget.Launch() API.")

        # Read the output file produced by running the program.
        with open(local_path, 'r') as f:
            output = f.read()

        self.expect(output, exe=False,
                    substrs=["a(1)", "b(2)", "a(3)"])

    def resolve_symbol_context_with_address(self):
        """Exercise SBTaget.ResolveSymbolContextForAddress() API."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create the two breakpoints inside function 'a'.
        breakpoint1 = target.BreakpointCreateByLocation('main.c', self.line1)
        breakpoint2 = target.BreakpointCreateByLocation('main.c', self.line2)
        #print("breakpoint1:", breakpoint1)
        #print("breakpoint2:", breakpoint2)
        self.assertTrue(breakpoint1 and
                        breakpoint1.GetNumLocations() == 1,
                        VALID_BREAKPOINT)
        self.assertTrue(breakpoint2 and
                        breakpoint2.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        #self.runCmd("process status")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line1)

        address1 = lineEntry.GetStartAddress()

        # Continue the inferior, the breakpoint 2 should be hit.
        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        #self.runCmd("process status")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line2)

        address2 = lineEntry.GetStartAddress()

        #print("address1:", address1)
        #print("address2:", address2)

        # Now call SBTarget.ResolveSymbolContextForAddress() with the addresses
        # from our line entry.
        context1 = target.ResolveSymbolContextForAddress(
            address1, lldb.eSymbolContextEverything)
        context2 = target.ResolveSymbolContextForAddress(
            address2, lldb.eSymbolContextEverything)

        self.assertTrue(context1 and context2)
        #print("context1:", context1)
        #print("context2:", context2)

        # Verify that the context point to the same function 'a'.
        symbol1 = context1.GetSymbol()
        symbol2 = context2.GetSymbol()
        self.assertTrue(symbol1 and symbol2)
        #print("symbol1:", symbol1)
        #print("symbol2:", symbol2)

        from lldbsuite.test.lldbutil import get_description
        desc1 = get_description(symbol1)
        desc2 = get_description(symbol2)
        self.assertTrue(desc1 and desc2 and desc1 == desc2,
                        "The two addresses should resolve to the same symbol")
