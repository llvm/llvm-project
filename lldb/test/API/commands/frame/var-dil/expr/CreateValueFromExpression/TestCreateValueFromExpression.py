"""
Test how DIL is used in SBValue::CreateValueFromExpression.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCreateValueFromExpression(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_formatter(self):
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        # Enable logging
        log_file = self.getBuildArtifact("log-file.txt")
        if os.path.exists(log_file):
            os.remove(log_file)
        self.runCmd("log enable -f '%s' lldb expr" % (log_file))

        self.runCmd("settings set target.experimental.use-DIL true")
        self.runCmd("settings set target.experimental.use-DIL-for-creating-values true")

        # Check expression results
        frame = thread.GetFrameAtIndex(0)
        i = frame.FindVariable("i")
        v1 = i.CreateValueFromExpression("v1", "i + 1")
        self.assertEqual(v1.GetValue(), "1")
        v2 = i.CreateValueFromExpression("v2", "static_cast<double>(i) + 2.5")
        self.assertEqual(v2.GetValue(), "2.5")
        self.runCmd(
            "settings set target.experimental.use-DIL-for-creating-values false"
        )
        v3 = i.CreateValueFromExpression("v3", "i + 3")
        self.assertEqual(v3.GetValue(), "3")

        with open(log_file, "r") as f:
            log = f.read()

        # Check that supported expression is evaluated by DIL
        self.assertGreater(log.find("v1 = 1 (evaluated by: DIL)"), 0)
        # Check that if DIL cannot evaluate the expression, it falls back to
        # full expression evaluation
        self.assertGreater(log.find("v2 = 2.5 (evaluated by: UserExpression)"), 0)
        # Check that creating values using DIL was disabled for the 3rd expression
        self.assertGreater(log.find("v3 = 3 (evaluated by: UserExpression)"), 0)
