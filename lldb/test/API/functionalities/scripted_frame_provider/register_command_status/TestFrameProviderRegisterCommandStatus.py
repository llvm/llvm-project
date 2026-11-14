"""
Test that `target frame-provider register` succeeds without asserting.

DoExecute never called CommandReturnObject::SetStatus() on its success
path. AppendMessage()/AppendMessageWithFormatv() don't touch the status
the way AppendError()/SetError() do, so it stayed eReturnStatusInvalid
and tripped CommandObject.cpp's DoExecuteStatusCheck assert.

Every other scripted_frame_provider test goes through
SBTarget::RegisterScriptedFrameProvider directly instead of this
command, which is how this slipped through.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFrameProviderRegisterCommandStatus(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_register_command_succeeds(self):
        """
        `target frame-provider register` should complete successfully
        (and not assert) when given a valid scripted frame provider class.
        """
        self.build()

        lldbutil.run_to_name_breakpoint(self, "frame3")

        provider_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + provider_path)

        self.expect(
            "target frame-provider register -C frame_provider.MinimalProvider",
            substrs=["successfully registered scripted frame provider"],
        )

    def test_static_method_exception_does_not_escape(self):
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid())

        provider_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + provider_path)
        self.runCmd(
            "target frame-provider register "
            "-C frame_provider.FailingDescriptionProvider"
        )

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "target frame-provider list", result
        )

        self.assertTrue(result.Succeeded(), result.GetError())
        self.assertIn("FailingDescriptionProvider", result.GetOutput())
