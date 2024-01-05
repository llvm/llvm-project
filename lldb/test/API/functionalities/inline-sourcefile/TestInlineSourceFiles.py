import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil


class InlineSourceFilesTestCase(TestBase):
    @skipIf(compiler="gcc")
    @skipIf(compiler="clang", compiler_version=["<", "18.0"])
    # dsymutil doesn't yet copy the sources
    @expectedFailureDarwin(debug_info=["dsym"])
    def test(self):
        """Test DWARF inline source files."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, 'f')
        self.expect("list f", substrs=["This is inline source code"])
