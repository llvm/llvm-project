"""Test the lldb public C++ api breakpoint callbacks.  """

import os, re, StringIO
import unittest2
from lldbtest import *
import lldbutil
import subprocess

class SBBreakpointCallbackCase(TestBase):

    mydir = os.path.join("api", "multithreaded")

    def setUp(self):
        TestBase.setUp(self)
        self.lib_dir = os.environ["LLDB_LIB_DIR"]
        self.inferior = 'inferior_program'
        if self.getArchitecture() != "i386":
          self.buildProgram('inferior.cpp', self.inferior)
          self.addTearDownHook(lambda: os.remove(self.inferior))

    @unittest2.expectedFailure # llvm.org/pr16000: SBBreakpoint.SetCallback() does nothing
    @skipIfi386
    @skipIfLinuxClang # buildbot clang version unable to use libstdc++ with c++11
    def test_breakpoint_callback(self):
        """Test the that SBBreakpoint callback is invoked when a breakpoint is hit. """
        self.build_and_test('driver.cpp test_breakpoint_callback.cpp',
                            'test_breakpoint_callback')

    @skipIfi386
    @skipIfLinuxClang # buildbot clang version unable to use libstdc++ with c++11
    def test_sb_api_listener_event_description(self):
        """ Test the description of an SBListener breakpoint event is valid."""
        self.build_and_test('driver.cpp listener_test.cpp test_listener_event_description.cpp',
                            'test_listener_event_description')
        pass

    @skipIfi386
    @skipIfLinuxClang # buildbot clang version unable to use libstdc++ with c++11
    def test_sb_api_listener_event_process_state(self):
        """ Test that a registered SBListener receives events when a process
            changes state.
        """
        self.build_and_test('driver.cpp listener_test.cpp test_listener_event_process_state.cpp',
                            'test_listener_event_process_state')
        pass


    @skipIfi386
    @skipIfLinuxClang # buildbot clang version unable to use libstdc++ with c++11
    @skipIfLinux # llvm.org/pr16016 assertion failure in ProcessPOSIX.cpp.
    def test_sb_api_listener_resume(self):
        """ Test that a process can be resumed from a non-main thread. """
        self.build_and_test('driver.cpp listener_test.cpp test_listener_resume.cpp',
                            'test_listener_resume')
        pass

    def build_and_test(self, sources, test_name, args = None):
        """ Build LLDB test from sources, and run expecting 0 exit code """
        self.buildDriver(sources, test_name)
        self.addTearDownHook(lambda: os.remove(test_name))

        exe = [os.path.join(os.getcwd(), test_name), self.inferior]

        if self.TraceOn():
            print "Running test %s" % " ".join(exe)

        check_call(exe, env={self.dylibPath : self.getLLDBLibraryEnvVal()})



    def build_program(self, sources, program):
        return self.buildDriver(sources, program)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
