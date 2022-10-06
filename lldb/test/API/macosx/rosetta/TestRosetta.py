import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


def get_os_version():
    try:
        os_version_str = subprocess.check_output(["sysctl", "kern.osversion"
                                                  ]).decode('utf-8')
    except subprocess.CalledProcessError:
        return None
    m = re.match(r'kern\.osversion: (\w+)', os_version_str)
    if m:
        return m.group(1)
    return None


def has_rosetta_shared_cache(os_version):
    if not os_version:
        return False
    macos_device_support = os.path.join(os.path.expanduser("~"), 'Library',
                                        'Developer', 'Xcode',
                                        'macOS DeviceSupport')
    for _, subdirs, _ in os.walk(macos_device_support):
        for subdir in subdirs:
            if os_version in subdir:
                return True
    return False


class TestRosetta(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessAppleSilicon
    @skipIfDarwinEmbedded
    def test_rosetta(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")

        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitWarning)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file)

        event = lldb.SBEvent()
        os_version = get_os_version()
        if not has_rosetta_shared_cache(os_version):
            self.assertTrue(listener.GetNextEvent(event))
        else:
            self.assertFalse(listener.GetNextEvent(event))
