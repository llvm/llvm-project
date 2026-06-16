import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# (symbol, function-pointer type) for each libc function in the test.
LIBC_FUNCTIONS = [
    ("memset", "void *(*)(void *, int, unsigned long)"),
    ("memcpy", "void *(*)(void *, const void *, unsigned long)"),
    ("memmove", "void *(*)(void *, const void *, unsigned long)"),
    ("memcmp", "int (*)(const void *, const void *, unsigned long)"),
]


class LibcCallsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfRemote
    def test_libc_funcs_do_not_resolve_to_dyld(self):
        """The JIT-resolved address of each libc function must not lie in
        the private dyld copies of the same name. Calling these functions from
        outside of dyld is not supported."""

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        # Check that dyld is actually loaded.
        loaded_module_names = [
            m.GetFileSpec().GetFilename() for m in self.target().module_iter()
        ]
        self.assertIn("dyld", loaded_module_names, "dyld must be loaded")

        for symbol, fn_type in LIBC_FUNCTIONS:
            with self.subTest(symbol=symbol):
                # Casting `<symbol>` through the spelled-out function-pointer
                # type gives clang a function type to bind to; the cast
                # result is the JIT-resolved address baked in by FindSymbol.
                result = self.frame().EvaluateExpression(f"(void *)({fn_type}){symbol}")
                self.assertSuccess(result.GetError())
                resolved_addr = result.GetValueAsUnsigned()
                self.assertNotEqual(resolved_addr, 0)

                resolved_module_name = (
                    self.target()
                    .ResolveLoadAddress(resolved_addr)
                    .GetModule()
                    .GetFileSpec()
                    .GetFilename()
                )
                self.assertNotEqual(
                    resolved_module_name, "dyld", f"Called dyld version of {symbol}"
                )

    def test_libc_calls_succeed(self):
        """Calling memset/memcpy/memmove/memcmp from expressions must
        execute without trapping."""

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        # Call libc functions should work without trapping.
        # We use a non-constant length so the compiler is not tempted to lower
        # this to a series of read/writes.
        self.expect_expr(
            """__builtin_memcpy(dst, src, len64);
            __builtin_memmove(dst, src, len64);
            __builtin_memset(dst, 0, len64);
            __builtin_memcmp(src, dst, len64)""",
            result_value="0",
        )
