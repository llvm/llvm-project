"""
Set the contents of variables and registers using raw data
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class SetDataTestCase(TestBase):

    mydir = os.path.join("functionalities", "set-data")

    @dsym_test
    def test_set_data_dsym(self):
        """Test setting the contents of variables and registers using raw data."""
        self.buildDsym()
        self.setData()

    def setData(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("br s -p First");
        self.runCmd("br s -p Second");

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("p myFoo.x", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['2'])

        process = self.dbg.GetSelectedTarget().GetProcess()
        frame = process.GetSelectedThread().GetFrameAtIndex(0)

        x = frame.FindVariable("myFoo").GetChildMemberWithName("x")

        my_data = lldb.SBData.CreateDataFromSInt32Array(lldb.eByteOrderLittle, 8, [4])
        err = lldb.SBError()

        self.assertTrue (x.SetData(my_data, err))

        self.runCmd("continue")

        self.expect("p myFoo.x", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['4'])

        frame = process.GetSelectedThread().GetFrameAtIndex(0)

        x = frame.FindVariable("string")

        if process.GetAddressByteSize() == 8:
            my_data = lldb.SBData.CreateDataFromUInt64Array(process.GetByteOrder(), 8, [0])
        else:
            my_data = lldb.SBData.CreateDataFromUInt32Array(process.GetByteOrder(), 4, [0])
        
        err = lldb.SBError()

        self.assertTrue (x.SetData(my_data, err))

        self.expect("fr var -d run-target string", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['NSString *', 'nil'])
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
