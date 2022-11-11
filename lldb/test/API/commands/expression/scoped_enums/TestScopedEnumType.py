

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ScopedEnumType(TestBase):

    @skipIf(dwarf_version=['<', '4'])
    def test(self):
        self.build()

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Set break point at this line.', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        self.expect("expr f == Foo::FooBar",
                substrs=['(bool) $0 = true'])

        self.expect_expr("f == Foo::FooBar", result_value='true')
        self.expect_expr("b == BarBar", result_value='true')

        ## b is not a Foo
        value = frame.EvaluateExpression("b == Foo::FooBar")
        self.assertTrue(value.IsValid())
        self.assertFalse(value.GetError().Success())

        ## integral is not implicitly convertible to a scoped enum
        value = frame.EvaluateExpression("1 == Foo::FooBar")
        self.assertTrue(value.IsValid())
        self.assertFalse(value.GetError().Success())
