"""
Tests that frame variable --depth and --element-count options work correctly
together
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestFrameVarDepthAndElemCount(TestBase):
    def test(self):
        """Test that bool types work in the expression parser"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        # Check that we print 5 elements but only 2 levels deep.
        self.expect(
            "frame var --depth 2 --element-count 5 -- c",
            substrs=[
                "[0] = {\n    b ={...}\n  }",
                "[1] = {\n    b ={...}\n  }",
                "[2] = {\n    b ={...}\n  }",
                "[3] = {\n    b ={...}\n  }",
                "[4] = {\n    b ={...}\n  }",
            ],
        )
