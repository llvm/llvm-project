"""Test that importing modules in C++ works as expected."""


import lldb
import shutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CXXModulesImportTestCase(TestBase):

    def build(self):
        include = self.getBuildArtifact('include')
        lldbutil.mkdir_p(include)
        for f in ['Foo.h', 'Bar.h', 'module.modulemap']:
            shutil.copyfile(self.getSourcePath(os.path.join('Inputs', f)),
                            os.path.join(include, f))
        super(CXXModulesImportTestCase, self).build()
    
    @skipUnlessDarwin
    @skipIf(macos_version=["<", "10.12"])
    @skipIf(compiler="clang", compiler_version=['<', '14.0'])
    def test_expr(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.cpp'))

        self.expect("expr -l Objective-C++ -- @import Bar")
        self.expect("expr -- Bar()", substrs = ["success"])
        self.expect("expr -l Objective-C++ -- @import THIS_MODULE_DOES_NOT_EXIST",
                    error=True)

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "10.12"])
    @skipIf(compiler="clang", compiler_version=['<', '14.0'])
    def test_expr_failing_import(self):
        self.build()
        shutil.rmtree(self.getBuildArtifact('include'))
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.cpp'))

        self.expect("expr -l Objective-C++ -- @import Bar", error=True)
