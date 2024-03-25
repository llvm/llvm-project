import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Test covariant return types for pointers to class that contains the called function.
        self.expect_expr("derived.getPtr()", result_type="Derived *")
        self.expect_expr("base_ptr_to_derived->getPtr()", result_type="Base *")
        self.expect_expr("base.getPtr()", result_type="Base *")
        # The same tests with reference types. LLDB drops the reference when it turns the
        # result into a SBValue so check for the underlying type of the result.
        self.expect_expr("derived.getRef()", result_type="Derived")
        self.expect_expr("base_ptr_to_derived->getRef()", result_type="Base")
        self.expect_expr("base.getRef()", result_type="Base")

        # Test covariant return types for pointers to class that does *not* contain the called function.
        self.expect_expr("derived.getOtherPtr()", result_type="OtherDerived *")
        self.expect_expr(
            "base_ptr_to_derived->getOtherPtr()", result_type="OtherBase *"
        )
        self.expect_expr("base.getOtherPtr()", result_type="OtherBase *")
        # The same tests with reference types. LLDB drops the reference when it turns the
        # result into a SBValue so check for the underlying type of the result.
        self.expect_expr("derived.getOtherRef()", result_type="OtherDerived")
        self.expect_expr("base_ptr_to_derived->getOtherRef()", result_type="OtherBase")
        self.expect_expr("base.getOtherRef()", result_type="OtherBase")

        # Test that we call the right function and get the right value back.
        self.expect_expr("derived.getOtherPtr()->value()", result_summary='"derived"')
        self.expect_expr(
            "base_ptr_to_derived->getOtherPtr()->value()", result_summary='"derived"'
        )
        self.expect_expr("base.getOtherPtr()->value()", result_summary='"base"')
        self.expect_expr("derived.getOtherRef().value()", result_summary='"derived"')
        self.expect_expr(
            "base_ptr_to_derived->getOtherRef().value()", result_summary='"derived"'
        )
        self.expect_expr("base.getOtherRef().value()", result_summary='"base"')

        self.expect_expr("referencing_derived.getOther()->get()->a", result_value="42")
