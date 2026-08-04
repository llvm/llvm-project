"""
Test lldb-dap locations request
"""

from lldbsuite.test.decorators import skipIf, skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_locations(DAPTestCaseBase):

    @skipIfWindows
    @skipIf(
        bugnumber="https://github.com/llvm/llvm-project/issues/203127", archs=["arm64e"]
    )
    def test_locations(self):
        """
        Tests the 'locations' request.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.cpp")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(
                source, [line_number(source, "break here")]
            )
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        top_frame = session.top_frame_from(stop_event)
        top_frame_locals = top_frame.locals

        # var1 has a declarationLocation but no valueLocation.
        var1 = top_frame_locals["var1"].variable
        decl_ref = self.expect_not_none(var1.declarationLocationReference)
        session.verify_location(decl_ref, "main.cpp", line_number(source, "var1 decl"))
        self.assertIsNone(var1.valueLocationReference)

        # func_ptr has both a declaration and a valueLocation.
        func_ptr = top_frame_locals["func_ptr"].variable
        decl_ref = self.expect_not_none(func_ptr.declarationLocationReference)
        session.verify_location(
            decl_ref, "main.cpp", line_number(source, "func_ptr decl")
        )
        value_ref = self.expect_not_none(func_ptr.valueLocationReference)
        session.verify_location(
            value_ref, "main.cpp", line_number(source, "greet decl")
        )

        # func_ref has both a declaration and a valueLocation.
        func_ref = top_frame_locals["func_ref"].variable
        decl_ref = self.expect_not_none(func_ref.declarationLocationReference)
        session.verify_location(
            decl_ref, "main.cpp", line_number(source, "func_ref decl")
        )
        value_ref = self.expect_not_none(func_ref.valueLocationReference)
        session.verify_location(
            value_ref, "main.cpp", line_number(source, "greet decl")
        )

        # member_ptr has both a declaration and a valueLocation.
        member_ptr = top_frame_locals["member_ptr"].variable
        decl_ref = self.expect_not_none(member_ptr.declarationLocationReference)
        session.verify_location(
            decl_ref, "main.cpp", line_number(source, "member_ptr decl")
        )
        value_ref = self.expect_not_none(member_ptr.valueLocationReference)
        session.verify_location(value_ref, "main.cpp", line_number(source, "foo decl"))

        # virtual_member_ptr has a declarationLocation but no valueLocation.
        virtual_member_ptr = top_frame_locals["virtual_member_ptr"].variable
        decl_ref = self.expect_not_none(virtual_member_ptr.declarationLocationReference)
        session.verify_location(
            decl_ref, "main.cpp", line_number(source, "virtual_member_ptr decl")
        )
        self.assertIsNone(virtual_member_ptr.valueLocationReference)

        # `evaluate` responses for function pointers also have locations associated.
        eval_body = top_frame.evaluate("greet")
        self.assertIsNotNone(eval_body.valueLocationReference)
