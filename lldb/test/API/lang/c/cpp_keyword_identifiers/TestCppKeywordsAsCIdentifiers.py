import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # FIXME: Clang on Windows somehow thinks static_assert is a C keyword.
    @skipIfWindows
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        # Test several variables with C++ keyword names and make sure they
        # work as intended in the expression parser.
        self.expect_expr("alignas", result_type="int", result_value="1")
        self.expect_expr("alignof", result_type="int", result_value="1")
        self.expect_expr("and", result_type="int", result_value="1")
        self.expect_expr("and_eq", result_type="int", result_value="1")
        self.expect_expr("atomic_cancel", result_type="int", result_value="1")
        self.expect_expr("atomic_commit", result_type="int", result_value="1")
        self.expect_expr("atomic_noexcept", result_type="int", result_value="1")
        self.expect_expr("bitand", result_type="int", result_value="1")
        self.expect_expr("bitor", result_type="int", result_value="1")
        self.expect_expr("catch", result_type="int", result_value="1")
        self.expect_expr("char8_t", result_type="int", result_value="1")
        self.expect_expr("char16_t", result_type="int", result_value="1")
        self.expect_expr("char32_t", result_type="int", result_value="1")
        self.expect_expr("class", result_type="int", result_value="1")
        self.expect_expr("compl", result_type="int", result_value="1")
        self.expect_expr("concept", result_type="int", result_value="1")
        self.expect_expr("consteval", result_type="int", result_value="1")
        self.expect_expr("constexpr", result_type="int", result_value="1")
        self.expect_expr("constinit", result_type="int", result_value="1")
        self.expect_expr("const_cast", result_type="int", result_value="1")
        self.expect_expr("co_await", result_type="int", result_value="1")
        self.expect_expr("co_return", result_type="int", result_value="1")
        self.expect_expr("co_yield", result_type="int", result_value="1")
        self.expect_expr("decltype", result_type="int", result_value="1")
        self.expect_expr("delete", result_type="int", result_value="1")
        self.expect_expr("dynamic_cast", result_type="int", result_value="1")
        self.expect_expr("explicit", result_type="int", result_value="1")
        self.expect_expr("export", result_type="int", result_value="1")
        self.expect_expr("friend", result_type="int", result_value="1")
        self.expect_expr("mutable", result_type="int", result_value="1")
        self.expect_expr("namespace", result_type="int", result_value="1")
        self.expect_expr("new", result_type="int", result_value="1")
        self.expect_expr("noexcept", result_type="int", result_value="1")
        self.expect_expr("not", result_type="int", result_value="1")
        self.expect_expr("not_eq", result_type="int", result_value="1")
        self.expect_expr("operator", result_type="int", result_value="1")
        self.expect_expr("or", result_type="int", result_value="1")
        self.expect_expr("or_eq", result_type="int", result_value="1")
        self.expect_expr("private", result_type="int", result_value="1")
        self.expect_expr("protected", result_type="int", result_value="1")
        self.expect_expr("public", result_type="int", result_value="1")
        self.expect_expr("reflexpr", result_type="int", result_value="1")
        self.expect_expr("reinterpret_cast", result_type="int", result_value="1")
        self.expect_expr("requires", result_type="int", result_value="1")
        self.expect_expr("static_assert", result_type="int", result_value="1")
        self.expect_expr("static_cast", result_type="int", result_value="1")
        self.expect_expr("synchronized", result_type="int", result_value="1")
        self.expect_expr("template", result_type="int", result_value="1")
        self.expect_expr("this", result_type="int", result_value="1")
        self.expect_expr("thread_local", result_type="int", result_value="1")
        self.expect_expr("throw", result_type="int", result_value="1")
        self.expect_expr("try", result_type="int", result_value="1")
        self.expect_expr("typeid", result_type="int", result_value="1")
        self.expect_expr("typename", result_type="int", result_value="1")
        self.expect_expr("virtual", result_type="int", result_value="1")
        self.expect_expr("xor", result_type="int", result_value="1")
        self.expect_expr("xor_eq", result_type="int", result_value="1")

        # Some keywords are not available in LLDB as their language feature
        # is enabled by default.

        # 'using' is used by LLDB for local variables.
        self.expect("expr using", error=True, substrs=["expected unqualified-id"])

        # 'wchar_t' supported is enabled in LLDB.
        self.expect("expr wchar_t", error=True, substrs=["expected unqualified-id"])

        # LLDB enables 'bool' support by default.
        self.expect("expr bool", error=True, substrs=["expected unqualified-id"])
        self.expect("expr false", error=True, substrs=["expected unqualified-id"])
        self.expect("expr true", error=True, substrs=["expected unqualified-id"])
