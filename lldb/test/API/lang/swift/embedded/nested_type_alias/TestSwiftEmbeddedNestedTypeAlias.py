import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedNestedTypeAlias(TestBase):
    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test(self):
        """Test rhat stored propertues typed through a nested type alias and
        used only as a generic argument is represented iun debug info."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        storage = thread.frames[0].FindVariable("storage")

        # `state` is the nested State struct. Computing its layout requires
        # resolving the type alias `Handler` used by its `handler` field;
        # without a DW_TAG_typedef for the alias this fails and State has no
        # children.
        state = storage.GetChildMemberWithName("state")
        lldbutil.check_variable(
            self, state, typename="a.Storage<Swift.Int>.State", num_children=2
        )

        handler = state.GetChildMemberWithName("handler")
        lldbutil.check_variable(
            self,
            handler,
            typename="Swift.Optional<@Sendable (Swift.Int) -> ()>",
            summary="nil",
        )

        flag = state.GetChildMemberWithName("flag")
        lldbutil.check_variable(self, flag, typename="Swift.Bool", summary="false")
