"""
Test some lldb command abbreviations.
"""
import commands
import lldb
import os
import time
import unittest2
from lldbtest import *
import lldbutil

def execute_command (command):
    #print '%% %s' % (command)
    (exit_status, output) = commands.getstatusoutput (command)
    #if output:
    #    print output
    #print 'status = %u' % (exit_status)
    return exit_status

class ExecTestCase(TestBase):

    mydir = os.path.join("functionalities", "exec")

        
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        if self.getArchitecture() == 'x86_64':
            source = os.path.join (os.getcwd(), "main.cpp")
            o_file = os.path.join (os.getcwd(), "main.o")
            execute_command ("'%s' -g -O0 -arch i386 -arch x86_64 '%s' -c -o '%s'" % (os.environ["CC"], source, o_file))
            execute_command ("'%s' -g -O0 -arch i386 -arch x86_64 '%s'" % (os.environ["CC"], o_file))
        else:
            self.buildDsym()
        self.do_test ()


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_with_dwarf (self):
        if self.getArchitecture() == 'x86_64':
            source = os.path.join (os.getcwd(), "main.cpp")
            o_file = os.path.join (os.getcwd(), "main.o")
            dsym_path = os.path.join (os.getcwd(), "a.out.dSYM")
            execute_command ("'%s' -g -O0 -arch i386 -arch x86_64 '%s' -c -o '%s'" % (os.environ["CC"], source, o_file))
            execute_command ("'%s' -g -O0 -arch i386 -arch x86_64 '%s'" % (os.environ["CC"], o_file))
            execute_command ("rm -rf '%s'" % (dsym_path))
        else:
            self.buildDwarf()
        self.do_test ()

    def do_test (self):
        exe = os.path.join (os.getcwd(), "a.out")
        
        # Create the target
        target = self.dbg.CreateTarget(exe)
        
        # Create any breakpoints we need
        breakpoint = target.BreakpointCreateBySourceRegex ('Set breakpoint 1 here', lldb.SBFileSpec ("main.cpp", False))
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)
        
        for i in range(6):
            # The stop reason of the thread should be breakpoint.
            self.assertTrue(process.GetState() == lldb.eStateStopped,
                            STOPPED_DUE_TO_BREAKPOINT)

            thread = process.GetThreadAtIndex (0)

            self.assertTrue (thread.IsValid(),
                             "Process stopped at 'main' should have a valid thread");

            stop_reason = thread.GetStopReason()
            
            self.assertTrue (stop_reason == lldb.eStopReasonBreakpoint,
                             "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint");

            # Run and we should stop due to exec
            process.Continue()
        
            self.assertTrue(process.GetState() == lldb.eStateStopped,
                            "Process should be stopped at __dyld_start")
                        
            thread = process.GetThreadAtIndex (0)
        
            self.assertTrue (thread.IsValid(),
                             "Process stopped at exec should have a valid thread");
        
            stop_reason = thread.GetStopReason()
        
            self.assertTrue (stop_reason == lldb.eStopReasonExec,
                             "Thread in process stopped on exec should have a stop reason of eStopReasonExec");
        
             # Run and we should stop at breakpoint in main after exec
            process.Continue()        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

