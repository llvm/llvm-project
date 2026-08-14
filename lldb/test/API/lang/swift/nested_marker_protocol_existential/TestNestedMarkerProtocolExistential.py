import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestNestedMarkerProtocolExistential(TestBase):
    @swiftTest
    @requireNotEmbeddedSwift
    def test(self):
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]

        # any Foo wrapping an Int: dynamic resolution lifts the payload to Int.
        lldbutil.check_variable(
            self, frame.FindVariable("bareMarker"), use_dynamic=True, value="1"
        )

        # any Foo & Bar wrapping Conformer(bar: 2).
        bareComposition = frame.FindVariable("bareComposition")
        lldbutil.check_variable(
            self, bareComposition.GetChildMemberWithName("bar"), value="2"
        )

        # [any Foo] = [5, 6, 7]: each element must lift to its Int payload.
        arrayOfMarker = frame.FindVariable("arrayOfMarker")
        for i, expected in enumerate(["5", "6", "7"]):
            lldbutil.check_variable(
                self,
                arrayOfMarker.GetChildAtIndex(i),
                use_dynamic=True,
                value=expected,
            )

        # Box<any Foo & Bar>(payload: Conformer(bar: 8), tag: 9).
        genericBox = frame.FindVariable("genericBox")
        lldbutil.check_variable(
            self,
            genericBox.GetChildMemberWithName("payload").GetChildMemberWithName(
                "bar"
            ),
            value="8",
        )
        lldbutil.check_variable(
            self, genericBox.GetChildMemberWithName("tag"), value="9"
        )
