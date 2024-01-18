# TestObjCIvarDiscovery.py
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
Test that we can correctly see ivars from the Objective-C runtime
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import re
import unittest2
import shutil

class TestObjCIVarDiscovery(TestBase):
    @skipUnlessDarwin
    @skipIf(debug_info=no_match("dsym"))
    @swiftTest
    def test_nodbg(self):
        self.build()
        shutil.rmtree(self.getBuildArtifact("aTestFramework.framework/Versions/A/aTestFramework.dSYM"))
        self.do_test(False)

    @skipUnlessDarwin
    @skipIf(debug_info=no_match("dsym"))
    @swiftTest
    def test_dbg(self):
        self.build()
        self.do_test(True)

    def prepare_value(self, value):
        value.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        value.SetPreferSyntheticValue(True)
        return value

    def do_test(self, dbg):
        """Test that we can correctly see ivars from the Objective-C runtime"""
        if lldb.remote_platform:
            wd = lldb.remote_platform.GetWorkingDirectory()
            directory = 'aTestFramework.framework/Versions/A/'
            filename = directory + '/aTestFramework'
            cur_dir = wd
            for d in directory.split('/'):
                err = lldb.remote_platform.MakeDirectory(
                    os.path.join(cur_dir, d))
                self.assertFalse(err.Fail(), 'Failed to mkdir ' + d + ':' + str(err))
                cur_dir = os.path.join(cur_dir, d)
            err = lldb.remote_platform.Put(
                lldb.SBFileSpec(self.getBuildArtifact(filename)),
                lldb.SBFileSpec(os.path.join(wd, filename)))
            self.assertFalse(err.Fail(), 'Failed to copy ' + filename + ':' + str(err))

        # Launch the process, and do not stop at the entry point.
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'),
        #    extra_images=['aTestFramework.framework/aTestFramework']
        )

        if dbg:
            self.expect(
                "image list", substrs=["Contents/Resources/DWARF/aTestFramework"])
        else:
            self.expect(
                "image list",
                substrs=["Contents/Resources/DWARF/aTestFramework"],
                matching=False)

        self.runCmd("frame variable -d run --show-types --ptr-depth=1")

        obj = self.prepare_value(self.frame().FindVariable("object"))

        mysubclass = self.prepare_value(obj.GetChildAtIndex(0))
        myclass = self.prepare_value(mysubclass.GetChildAtIndex(0))

        m_pair = myclass.GetChildMemberWithName("m_pair")
        m_pair_A = m_pair.GetChildMemberWithName("A")
        m_pair_B = m_pair.GetChildMemberWithName("B")

        self.assertEqual(m_pair_A.GetValueAsUnsigned(), 1)
        self.assertEqual(m_pair_B.GetValueAsUnsigned(), 2)

        m_derived = self.prepare_value(
            myclass.GetChildMemberWithName("m_base"))

        m_derivedX = m_derived.GetChildMemberWithName("m_DerivedX")

        self.assertEqual(m_derivedX.GetValueAsUnsigned(), 1)

        m_numbers = self.prepare_value(
            myclass.GetChildMemberWithName("m_myclass_numbers"))

        self.assertTrue(
            m_numbers.GetSummary() == '3 elements',
            "m_myclass_numbers != 3 elements")

        m_subclass_ivar = mysubclass.GetChildMemberWithName("m_subclass_ivar")
        self.assertTrue(
            m_subclass_ivar.GetValueAsUnsigned() == 42,
            "m_subclass_ivar != 42")

        m_mysubclass_s = mysubclass.GetChildMemberWithName("m_mysubclass_s")
        self.assertTrue(
            m_mysubclass_s.GetSummary() == '"an NSString here"',
            'm_subclass_s != "an NSString here"')

        swiftivar = obj.GetChildMemberWithName("swiftivar")
        self.assertTrue(
            swiftivar.GetSummary() == '"Hey Swift!"', "swiftivar != Hey Swift")

        silly = self.prepare_value(obj.GetChildMemberWithName("silly"))

        silly_x = silly.GetChildMemberWithName("x")
        silly_url = silly.GetChildMemberWithName("url")

        self.assertTrue(silly_x.GetValueAsUnsigned() == 12, "x != 12")
        self.assertTrue(
            silly_url.GetSummary() == '"http://www.apple.com"',
            "url != apple.com")
