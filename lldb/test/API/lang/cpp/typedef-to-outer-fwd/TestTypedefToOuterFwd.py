import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCaseTypedefToOuterFwd(TestBase):
    '''
    We are stopped in main.o, which only sees a forward declaration
    of FooImpl. We then try to get the FooImpl::Ref typedef (whose
    definition is in lib.o). Make sure we correctly resolve this
    typedef.
    '''
    def test(self):
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "return", lldb.SBFileSpec("main.cpp")
        )

        foo = thread.frames[0].FindVariable('foo')
        self.assertSuccess(foo.GetError(), "Found foo")

        foo_type = foo.GetType()
        self.assertTrue(foo_type)

        impl = foo_type.GetPointeeType()
        self.assertTrue(impl)

        ref = impl.FindDirectNestedType('Ref')
        self.assertTrue(ref)

        self.assertEqual(ref.GetCanonicalType(), foo_type)
