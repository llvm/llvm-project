import contextlib
import os
import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def haswell():
    features = subprocess.check_output(["sysctl", "machdep.cpu"])
    return "AVX2" in features.decode('utf-8')


def apple_silicon():
    features = subprocess.check_output(["sysctl", "machdep.cpu"])
    return "Apple M" in features.decode('utf-8')


@contextlib.contextmanager
def remove_from_env(var):
    old_environ = os.environ.copy()
    del os.environ[var]
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class TestLaunchProcessPosixSpawn(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    def no_haswell(self):
        if not haswell():
            return "Current CPU is not Haswell"
        return None

    def no_apple_silicon(self):
        if not apple_silicon():
            return "Current CPU is not Apple Silicon"
        return None

    def run_arch(self, exe, arch):
        self.runCmd('target create -arch {} {}'.format(arch, exe))
        self.runCmd('run')

        process = self.dbg.GetSelectedTarget().process
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertIn('slice: {}'.format(arch), process.GetSTDOUT(1000))

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipTestIfFn(no_haswell)
    def test_haswell(self):
        self.build()
        exe = self.getBuildArtifact("fat.out")
        self.run_arch(exe, 'x86_64')
        self.run_arch(exe, 'x86_64h')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipTestIfFn(no_apple_silicon)
    def test_apple_silicon(self):
        self.build()
        exe = self.getBuildArtifact("fat.out")

        # We need to remove LLDB_DEBUGSERVER_PATH from the environment if it's
        # set so that the Rosetta debugserver is picked for x86_64.
        with remove_from_env('LLDB_DEBUGSERVER_PATH'):
            self.run_arch(exe, 'x86_64')
            self.run_arch(exe, 'arm64')
