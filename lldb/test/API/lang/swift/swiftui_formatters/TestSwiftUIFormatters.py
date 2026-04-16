import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skip(bugnumber="rdar://174869708")
@skipIf(archs=["x86_64"], bugnumber="rdar://174750739")
class TestCase(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test_before(self):
        self.build()
        _, _, self.thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break before", lldb.SBFileSpec("main.swift")
        )
        self._do_test("view._count", 41, is_graph_update=False)

    @skipUnlessDarwin
    @swiftTest
    def test_body(self):
        self.build()
        _, _, self.thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break body", lldb.SBFileSpec("main.swift")
        )
        self._do_test("self._count", 41, is_graph_update=True)

    @skipUnlessDarwin
    @swiftTest
    def test_after(self):
        self.build()
        log = self.getBuildArtifact("types.log")
        self.expect(f"log enable lldb types -v -f {log}")
        _, _, self.thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break after", lldb.SBFileSpec("main.swift")
        )
        self._do_test("self._count", 15, is_graph_update=False)

    @skipUnlessDarwin
    @swiftTest
    def test_final(self):
        self.build()
        log = self.getBuildArtifact("types.log")
        self.expect(f"log enable lldb types -v -f {log}")
        _, _, self.thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break final", lldb.SBFileSpec("main.swift")
        )
        self._do_test("self._count", 23, is_graph_update=False)

    def _do_test(self, var_name: str, value: int, *, is_graph_update: bool):
        symbol = "AG::Graph::UpdateStack::update()"
        if is_graph_update:
            self.assertIn(symbol, (f.name for f in self.thread))
        else:
            self.assertNotIn(symbol, (f.name for f in self.thread))

        frame = self.thread.selected_frame
        count = frame.var(var_name)
        self.assertEqual(count.GetNumChildren(), 1)
        self.assertEqual(count.member["wrappedValue"].unsigned, value)
        self.assertEqual(count.summary, str(value))
