"""Test the lldb public C++ api for returning SBCommandReturnObject."""

from __future__ import print_function


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSBCommandReturnObject(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNoSBHeaders
    @expectedFailureAll(
        oslist=["windows"], archs=["i[3-6]86", "x86_64"],
        bugnumber="llvm.org/pr43570")
    def test_sb_command_return_object(self):
        env = {self.dylibPath: self.getLLDBLibraryEnvVal()}

        self.driver_exe = self.getBuildArtifact("command-return-object")
        self.buildDriver('main.cpp', self.driver_exe)
        self.addTearDownHook(lambda: os.remove(self.driver_exe))
        self.signBinary(self.driver_exe)

        if self.TraceOn():
            print("Running test %s" % self.driver_exe)
            check_call([self.driver_exe, self.driver_exe], env=env)
        else:
            with open(os.devnull, 'w') as fnull:
                check_call([self.driver_exe, self.driver_exe],
                           env=env, stdout=fnull, stderr=fnull)
