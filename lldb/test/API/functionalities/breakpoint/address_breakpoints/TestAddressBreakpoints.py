"""
Test address breakpoints set with shared library of SBAddress work correctly.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class AddressBreakpointTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_address_breakpoints(self):
        """Test address breakpoints set with shared library of SBAddress work correctly."""
        self.build()
        self.address_breakpoints()

    def address_breakpoints(self):
        """Test address breakpoints set with shared library of SBAddress work correctly."""
        target = self.createTestTarget()

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Next get the address from the location, and create an address breakpoint using
        # that address:

        address = location.GetAddress()
        target.BreakpointDelete(breakpoint.GetID())

        breakpoint = target.BreakpointCreateBySBAddress(address)

        # Disable ASLR.  This will allow us to actually test (on platforms that support this flag)
        # that the breakpoint was able to track the module.

        launch_info = lldb.SBLaunchInfo(None)
        flags = launch_info.GetLaunchFlags()
        flags &= ~lldb.eLaunchFlagDisableASLR
        flags &= lldb.eLaunchFlagInheritTCCFromParent
        launch_info.SetLaunchFlags(flags)

        error = lldb.SBError()

        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should be 1.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        process.Kill()

        # Now re-launch and see that we hit the breakpoint again:
        launch_info.Clear()
        launch_info.SetLaunchFlags(flags)

        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should now be 2.
        self.assertEquals(breakpoint.GetHitCount(), 2)
