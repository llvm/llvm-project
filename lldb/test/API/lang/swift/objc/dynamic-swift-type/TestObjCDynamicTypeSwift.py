import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @skipUnlessFoundation
    @swiftTest
    def test(self):
        """Verify printing of Swift implemented ObjC objects."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))

        # For Swift implemented objects, it's assumed the ObjC runtime prints
        # only a pointer, and cannot generate any child values.
        self.expect("v -d no app", startstr="(App *) app = 0x")
        self.expect(
            "v -d no -P1 app",
            matching=False,
            substrs=["name", "version", "recentDocuments"],
        )

        # With dynamic typing, the Swift runtime produces Swift child values.
        self.expect("v app", substrs=["App?) app = 0x"])
        self.expect("v app.name", startstr='(String) app.name = "Debugger"')
        self.expect(
            "v app.version", startstr="((Int, Int)) app.version = (0 = 1, 1 = 0)"
        )

        documents = """\
([a.Document]?) app.recentDocuments = 1 value {
  [0] = {
    kind = binary
    path = "/path/to/something"
  }
}
"""
        self.expect("v app.recentDocuments", startstr=documents)
