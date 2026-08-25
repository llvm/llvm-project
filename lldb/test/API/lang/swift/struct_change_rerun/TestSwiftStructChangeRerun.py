# TestSwiftStructChangeRerun.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that we display self correctly for an inline-initialized struct
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import shutil


class TestSwiftStructChangeRerun(TestBase):
    SHARED_BUILD_TESTCASE = False

    @expectedFailureWindows  # https://github.com/swiftlang/llvm-project/issues/13444
    @swiftTest
    @skipIf(
        oslist=["windows"],
        swift_module_importer="noclang",
        bugnumber="rdar://178182243",
    )
    def test_swift_struct_change_rerun(self):
        """Test that we display self correctly for an inline-initialized struct"""
        copied_main_swift = self.getBuildArtifact("main.swift")
        
        # Cleanup the copied source file
        def cleanup():
            if os.path.exists(copied_main_swift):
                os.unlink(copied_main_swift)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        print('build with main1.swift')
        cleanup()
        shutil.copyfile("main1.swift", copied_main_swift)
        self.build()
        (target, process, thread, breakpoint) = \
            lldbutil.run_to_source_breakpoint(
                self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        var_a = self.frame().EvaluateExpression("a")
        var_a_a = var_a.GetChildMemberWithName("a")
        lldbutil.check_variable(self, var_a_a, False, value="12")

        var_a_b = var_a.GetChildMemberWithName("b")
        lldbutil.check_variable(self, var_a_b, False, '"Hey"')

        var_a_c = var_a.GetChildMemberWithName("c")
        self.assertFalse(var_a_c.IsValid(), "make sure a.c doesn't exist")
        process.Kill()

        # The updated copied_main_swift might not get rebuilt by Make if it is
        # changed within the same second, as they will have the same mtime.
        # A clean build fixes this.
        self.build(make_targets=["clean"])
        
        print('build with main2.swift')
        cleanup()
        shutil.copyfile("main2.swift", copied_main_swift)
        self.build()

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)
        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)

        var_a = self.frame().EvaluateExpression("a")
        var_a_a = var_a.GetChildMemberWithName("a")
        lldbutil.check_variable(self, var_a_a, False, value="12")

        var_a_b = var_a.GetChildMemberWithName("b")
        lldbutil.check_variable(self, var_a_b, False, '"Hey"')

        var_a_c = var_a.GetChildMemberWithName("c")
        self.assertTrue(var_a_c.IsValid(), "make sure a.c does exist")
        lldbutil.check_variable(self, var_a_c, False, value='12.125')

