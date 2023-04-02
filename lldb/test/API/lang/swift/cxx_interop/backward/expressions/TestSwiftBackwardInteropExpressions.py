
"""
Test that evaluating expressions works on backward interop mode.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropExpressions(TestBase):

    @skipIfLinux
    @swiftTest
    def test_func_step_in(self):
        self.build()
        self.runCmd('setting set target.experimental.swift-enable-cxx-interop true')
        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, 'Break here', lldb.SBFileSpec('main.cpp'))
        self.expect('expr swiftFunc()', substrs=["Inside a Swift function"])
        self.expect('expr swiftClass.swiftMethod()', substrs=["Inside a Swift method"])
        self.expect('expr a::SwiftClass::swiftStaticMethod()', substrs=["In a Swift static method"])
        self.expect('expr swiftClass.getSwiftProperty()', substrs=["This is a class with properties"])
        self.expect('expr swiftSubclassAsClass.overrideableMethod()', substrs=["In subclass"])
