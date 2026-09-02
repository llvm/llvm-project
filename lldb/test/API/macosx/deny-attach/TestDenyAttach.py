import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DenyAttachTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfDarwinEmbedded  # PT_DENY_ATTACH attach behavior differs on ios/tvos/etc
    @skipIfAsan  # Attach tests time out inconsistently under asan.
    def test_attach_to_deny_attach_process(self):
        """Attaching to a PT_DENY_ATTACH process reports an error, not a crash."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Use a file as a synchronization point between test and inferior: the
        # inferior writes its pid only after it has called PT_DENY_ATTACH.
        pid_file_path = lldbutil.append_to_process_working_directory(
            self, "pid_file_%d" % (int(time.time()))
        )
        self.addTearDownHook(
            lambda: self.run_platform_command("rm %s" % (pid_file_path))
        )

        popen = self.spawnSubprocess(exe, [pid_file_path])
        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        self.expect(
            "process attach -p " + pid,
            startstr="error: attach failed:",
            substrs=["PT_DENY_ATTACH"],
            error=True,
        )
