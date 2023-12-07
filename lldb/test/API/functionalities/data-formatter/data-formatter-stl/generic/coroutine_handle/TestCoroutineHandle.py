"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

USE_LIBSTDCPP = "USE_LIBSTDCPP"
USE_LIBCPP = "USE_LIBCPP"


class TestCoroutineHandle(TestBase):
    def do_test(self, stdlib_type):
        """Test std::coroutine_handle is displayed correctly."""
        self.build(dictionary={stdlib_type: "1"})
        is_clang = self.expectedCompiler(["clang"])

        test_generator_func_ptr_re = re.compile(
            r"^\(a.out`my_generator_func\(\) at main.cpp:[0-9]*\)$"
        )

        # Run until the initial suspension point
        lldbutil.run_to_source_breakpoint(
            self, "// Break at initial_suspend", lldb.SBFileSpec("main.cpp", False)
        )

        if self.frame().FindVariable("is_supported").GetValueAsUnsigned(1) == 0:
            self.skipTest("c++ library not supported")

        # Check that we show the correct function pointers and the `promise`.
        self.expect_expr(
            "gen.hdl",
            result_summary=re.compile("^coro frame = 0x[0-9a-f]*$"),
            result_children=[
                ValueCheck(name="resume", summary=test_generator_func_ptr_re),
                ValueCheck(name="destroy", summary=test_generator_func_ptr_re),
                ValueCheck(
                    name="promise",
                    children=[
                        ValueCheck(name="current_value", value="-1"),
                    ],
                ),
            ],
        )
        if is_clang:
            # For a type-erased `coroutine_handle<>`, we can still devirtualize
            # the promise call and display the correctly typed promise.
            self.expect_expr(
                "type_erased_hdl",
                result_summary=re.compile("^coro frame = 0x[0-9a-f]*$"),
                result_children=[
                    ValueCheck(name="resume", summary=test_generator_func_ptr_re),
                    ValueCheck(name="destroy", summary=test_generator_func_ptr_re),
                    ValueCheck(
                        name="promise",
                        children=[
                            ValueCheck(name="current_value", value="-1"),
                        ],
                    ),
                ],
            )
            # For an incorrectly typed `coroutine_handle`, we use the user-supplied
            # incorrect type instead of inferring the correct type. Strictly speaking,
            # incorrectly typed coroutine handles are undefined behavior. However,
            # it provides probably a better debugging experience if we display the
            # promise as seen by the program instead of fixing this bug based on
            # the available debug info.
            self.expect_expr(
                "incorrectly_typed_hdl",
                result_summary=re.compile("^coro frame = 0x[0-9a-f]*$"),
                result_children=[
                    ValueCheck(name="resume", summary=test_generator_func_ptr_re),
                    ValueCheck(name="destroy", summary=test_generator_func_ptr_re),
                    ValueCheck(name="promise", dereference=ValueCheck(value="-1")),
                ],
            )

        process = self.process()

        # Break at a coroutine body
        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "int_generator my_generator_func",
            lldb.SBFileSpec("main.cpp", False),
        )

        # Expect artificial variables to be displayed
        self.expect("frame variable", substrs=["__promise", "__coro_frame"])

        # Run until after the `co_yield`
        lldbutil.continue_to_source_breakpoint(
            self, process, "// Break after co_yield", lldb.SBFileSpec("main.cpp", False)
        )
        # We correctly show the updated value inside `promise.current_value`.
        self.expect_expr(
            "gen.hdl",
            result_children=[
                ValueCheck(name="resume", summary=test_generator_func_ptr_re),
                ValueCheck(name="destroy", summary=test_generator_func_ptr_re),
                ValueCheck(
                    name="promise",
                    children=[
                        ValueCheck(name="current_value", value="42"),
                    ],
                ),
            ],
        )

        # Run until the `final_suspend`
        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "// Break at final_suspend",
            lldb.SBFileSpec("main.cpp", False),
        )
        # At the final suspension point, `resume` is set to a nullptr.
        # Check that we still show the remaining data correctly.
        self.expect_expr(
            "gen.hdl",
            result_children=[
                ValueCheck(name="resume", value=re.compile("^0x0+$")),
                ValueCheck(name="destroy", summary=test_generator_func_ptr_re),
                ValueCheck(
                    name="promise",
                    children=[
                        ValueCheck(name="current_value", value="42"),
                    ],
                ),
            ],
        )
        if is_clang:
            # Devirtualization still works, also at the final suspension point, despite
            # the `resume` pointer being reset to a nullptr
            self.expect_expr(
                "type_erased_hdl",
                result_summary=re.compile("^coro frame = 0x[0-9a-f]*$"),
                result_children=[
                    ValueCheck(name="resume", value=re.compile("^0x0+$")),
                    ValueCheck(name="destroy", summary=test_generator_func_ptr_re),
                    ValueCheck(
                        name="promise",
                        children=[
                            ValueCheck(name="current_value", value="42"),
                        ],
                    ),
                ],
            )

    @add_test_categories(["libstdcxx"])
    def test_libstdcpp(self):
        self.do_test(USE_LIBSTDCPP)

    @add_test_categories(["libc++"])
    @skipIf(compiler="clang", compiler_version=["<", "15.0"])
    def test_libcpp(self):
        self.do_test(USE_LIBCPP)
