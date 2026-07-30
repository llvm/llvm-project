import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFunctionNameWithoutArgs(TestBase):
    @skipUnlessDarwin
    @no_debug_info_test
    def test_function_name_without_args(self):
        self.build()
        target = self.createTestTarget()
        target.LaunchSimple(None, None, self.get_process_working_directory())

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect(
            "bt",
            substrs=[
                "stop reason = hit program assert",
                "libsystem_kernel.dylib`__pthread_kill",
            ],
        )
        self.runCmd(
            'settings set frame-format "frame #${frame.index}: ${function.name-without-args}\n"'
        )
        self.expect(
            "bt",
            substrs=["stop reason = hit program assert", "frame #0: __pthread_kill"],
        )
