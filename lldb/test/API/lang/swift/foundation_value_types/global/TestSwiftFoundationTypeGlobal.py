import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftFoundationValueTypeGlobal(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    @skipUnlessFoundation
    def test(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact())
        self.assertTrue(target, VALID_TARGET)

        # Target variable without a process.
        # This is not actually expected to work, but also shouldn't crash.
        self.expect("target variable g_url")
