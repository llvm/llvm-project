import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestRecognizeBreakpoint(GDBRemoteTestBase):

    mydir = TestBase.compute_mydir(__file__)

    """ This tests the case where the gdb-remote server doesn't support any
        of the thread-info packets, and just tells which thread got the stop
        signal with:
              T05thread:01;
        There was a bug in lldb that we would set the stop reason from this 
        packet too early - before we had updated the thread list.  So when we
        later updated the thread list, we would throw away this info.  Normally
        we would be able to reconstruct it from the thread info, but not if the
        stub doesn't support it """
             
    @skipIfXmlSupportMissing
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.thread_info_count = 0
                self.after_cont = False
                self.current_thread = 0
                
            def cont(self):
                # Simulate process stopping due to a breakpoint:
                self.after_cont = True
                return "T05thread:01;"

            def vCont(self, packet):
                self.after_cont = True
                return "T05thread:01;"
            
            def haltReason(self):
                return "T02thread:01;"

            def threadStopInfo(self, num):
                return ""

            def QThreadSuffixSupported(self):
                return ""

            def QListThreadsInStopReply(self):
                return ""

            def setBreakpoint(self, packet):
                return "OK"
            
            def qfThreadInfo(self):
                return "m1"

            def qsThreadInfo(self):
                if (self.thread_info_count % 2) == 0:
                    str = "m2"
                else:
                    str = "l"
                self.thread_info_count += 1
                return str

            def readRegisters(self):
                if self.after_cont and self.current_thread == 1:
                    return "c01e990080ffffff"
                else:
                    return "badcfe10325476980"
            
            def readRegister(self, regno):
                return ""
            
            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                          </feature>
                        </target>""", False
                else:
                    return None, False

            def selectThread(self, op, thread):
                if op != 'g':
                    return ''
                
                self.current_thread = thread
                return "OK"
            
            def other (self, packet):
                if packet == "vCont?":
                    return "vCont;c;C;s;S"
                return ''
                
        python_os_plugin_path = os.path.join(self.getSourceDir(),
                                             'operating_system_2.py')
        command ="settings set target.process.python-os-plugin-path '{}'".format(
            python_os_plugin_path)
        self.runCmd(command)

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        bkpt = target.BreakpointCreateByAddress(0xffffff8000991ec0)
        self.assertEqual(bkpt.GetNumLocations(), 1, "Fake breakpoint was resolved.")

        # Get the initial stop, and we should have two threads.
        num_threads = len(process.threads)
        self.assertEqual(num_threads, 2, "Got two threads")

        thread_0 = process.threads[0]
        self.assertEqual(thread_0.GetStopReason(), 1, "Thread_0 stopped for no reason")
        self.assertEqual(thread_0.GetName(), "one", "Thread_0 is called one")
        
        thread_1 = process.threads[1]
        self.assertEqual(thread_1.GetStopReason(), 5, "Thread_0 stopped for SIGSTOP")
        self.assertEqual(thread_1.GetName(), "two", "Thread_0 is called two")
        
        # Now continue and we will fake hitting a breakpoint.
        process.Continue()

        self.assertEqual(process.GetState(),lldb.eStateStopped, "Process is stopped")
        num_threads = len(process.threads)

        num_threads = len(process.threads)
        self.assertEqual(num_threads, 2, "Got two threads")

        thread_0 = process.threads[0]
        self.assertEqual(thread_0.GetStopReason(), 1, "Thread_0 stopped for no reason")
        self.assertEqual(thread_0.GetName(), "one", "Thread_0 is called one")
        
        thread_1 = process.threads[1]
        self.assertEqual(thread_1.GetStopReason(), 3, "Thread_0 stopped for SIGTRAP")
        self.assertEqual(thread_1.GetName(), "three", "Thread_0 is called three")

        self.assertTrue(thread_1.IsValid(), "Thread_1 is valid")
        self.assertEqual(thread_1.GetStopReason(), lldb.eStopReasonBreakpoint, "Stopped at breakpoint")
        
