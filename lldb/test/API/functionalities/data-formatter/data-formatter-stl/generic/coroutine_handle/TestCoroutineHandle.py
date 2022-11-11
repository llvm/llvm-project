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

        test_generator_func_ptr_re = re.compile(
                r"^\(a.out`my_generator_func\(\) at main.cpp:[0-9]*\)$")

        # Run until the initial suspension point
        lldbutil.run_to_source_breakpoint(self, '// Break at initial_suspend',
                lldb.SBFileSpec("main.cpp", False))

        if self.frame().FindVariable("is_supported").GetValueAsUnsigned(1) == 0:
            self.skipTest("c++ library not supported")

        # Check that we show the correct function pointers and the `promise`. 
        self.expect_expr("gen.hdl",
            result_summary=re.compile("^coro frame = 0x[0-9a-f]*$"),
            result_children=[
                ValueCheck(name="resume", summary = test_generator_func_ptr_re),
                ValueCheck(name="destroy", summary = test_generator_func_ptr_re),
                ValueCheck(name="promise", children=[
                    ValueCheck(name="current_value", value = "-1"),
                ])
            ])
        # For type-erased `coroutine_handle<>` we are missing the `promise`
        # but still show `resume` and `destroy`.
        self.expect_expr("type_erased_hdl",
            result_summary=re.compile("^coro frame = 0x[0-9a-f]*$"),
            result_children=[
                ValueCheck(name="resume", summary = test_generator_func_ptr_re),
                ValueCheck(name="destroy", summary = test_generator_func_ptr_re),
            ])

        # Run until after the `co_yield`
        process = self.process()
        lldbutil.continue_to_source_breakpoint(self, process,
                '// Break after co_yield', lldb.SBFileSpec("main.cpp", False))
        # We correctly show the updated value inside `prommise.current_value`.
        self.expect_expr("gen.hdl",
            result_children=[
                ValueCheck(name="resume", summary = test_generator_func_ptr_re),
                ValueCheck(name="destroy", summary = test_generator_func_ptr_re),
                ValueCheck(name="promise", children=[
                    ValueCheck(name="current_value", value = "42"),
                ])
            ])
        
        # Run until the `final_suspend`
        lldbutil.continue_to_source_breakpoint(self, process,
                '// Break at final_suspend', lldb.SBFileSpec("main.cpp", False))
        # At the final suspension point, `resume` is set to a nullptr.
        # Check that we still show the remaining data correctly.
        self.expect_expr("gen.hdl",
            result_children=[
                ValueCheck(name="resume", value = re.compile("^0x0+$")),
                ValueCheck(name="destroy", summary = test_generator_func_ptr_re),
                ValueCheck(name="promise", children=[
                    ValueCheck(name="current_value", value = "42"),
                ])
            ])

    @add_test_categories(["libstdcxx"])
    def test_libstdcpp(self):
        self.do_test(USE_LIBSTDCPP)

    @add_test_categories(["libc++"])
    @skipIf(compiler="clang", compiler_version=['<', '15.0'])
    def test_libcpp(self):
        self.do_test(USE_LIBCPP)
