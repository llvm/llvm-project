import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class OdrHandlingWithDylibTestCase(TestBase):
    def test(self):
        """
        TODO
        """
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "plugin_entry", lldb.SBFileSpec("plugin.cpp")
        )

        self.expect_expr("*gProxyThis")
