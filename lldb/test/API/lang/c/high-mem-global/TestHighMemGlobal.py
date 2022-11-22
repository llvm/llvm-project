"""Look that lldb can display a global loaded in high memory at an addressable address."""


import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *

class TestHighMemGlobal(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin  # hardcoding of __DATA segment name
    def test_command_line(self):
        """Test that we can display a global variable loaded in high memory."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        err = lldb.SBError()

        target = self.dbg.CreateTarget(exe, '', '', False, err)
        self.assertTrue(target.IsValid())
        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())
        data_segment = module.FindSection("__DATA")
        self.assertTrue(data_segment.IsValid())
        err.Clear()

        self.expect("expr -- global.c", substrs=[' = 1'])
        self.expect("expr -- global.d", substrs=[' = 2'])
        self.expect("expr -- global.e", substrs=[' = 3'])

        err = target.SetSectionLoadAddress(data_segment, 0xffffffff00000000)
        self.assertTrue(err.Success())
        self.expect("expr -- global.c", substrs=[' = 1'])
        self.expect("expr -- global.d", substrs=[' = 2'])
        self.expect("expr -- global.e", substrs=[' = 3'])

        err = target.SetSectionLoadAddress(data_segment, 0x0000088100004000)
        self.assertTrue(err.Success())
        self.expect("expr -- global.c", substrs=[' = 1'])
        self.expect("expr -- global.d", substrs=[' = 2'])
        self.expect("expr -- global.e", substrs=[' = 3'])

        # This is an address in IRMemoryMap::FindSpace where it has an 
        # lldb-side buffer of memory that's used in IR interpreters when
        # memory cannot be allocated in the inferior / functions cannot
        # be jitted.
        err = target.SetSectionLoadAddress(data_segment, 0xdead0fff00000000)
        self.assertTrue(err.Success())

        # The global variable `global` is now overlayed by this 
        # IRMemoryMap special buffer, and now we cannot see the variable.
        # Testing that we get the incorrect values at this address ensures 
        # that IRMemoryMap::FindSpace and this test stay in sync.
        self.runCmd("expr -- int $global_c = global.c")
        self.runCmd("expr -- int $global_d = global.d")
        self.runCmd("expr -- int $global_e = global.e")
        self.expect("expr -- $global_c != 1 || $global_d != 2 || $global_e != 3", substrs=[' = true'])
