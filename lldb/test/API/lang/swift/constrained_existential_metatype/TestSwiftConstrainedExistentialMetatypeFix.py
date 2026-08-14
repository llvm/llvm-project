import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftConstrainedExistentialMetatypeFix(TestBase):
    @swiftTest
    @requireNotEmbeddedSwift
    def test(self):
        self.build()

        types_log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -v -f "%s"' % types_log)
        self.runCmd("settings set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.GetSelectedFrame()
        var = frame.FindVariable("variable")
        self.assertTrue(var.IsValid(), "variable not found in frame")
        static = var.GetStaticValue()
        self.assertTrue(
            static.GetError().Success(),
            "static existential-metatype value must lower without error",
        )
        lldbutil.check_variable(
            self,
            static,
            typename="@thick any a.MyProto<Self.T == Swift.Int>.Type",
            num_children=0,
        )
        lldbutil.check_variable(
            self,
            var,
            use_dynamic=True,
            typename="a.MyImpl.Type",
            summary="a.MyImpl",
        )

        with open(types_log) as f:
            log = f.read()
        self.assertNotIn("invalid existential metatype", log)
        self.assertNotIn("Couldn't compute size of type", log)
