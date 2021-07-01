import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(bugnumber="llvm.org/pr50814", compiler="gcc")
    def test_constructors(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,"// break here", lldb.SBFileSpec("main.cpp"))
        self.expect_expr("ClassWithImplicitCtor().foo()", result_type="int", result_value="1")
        self.expect_expr("ClassWithOneCtor(2).value", result_type="int", result_value="2")
        self.expect_expr("ClassWithMultipleCtor(3).value", result_type="int", result_value="3")
        self.expect_expr("ClassWithMultipleCtor(3, 1).value", result_type="int", result_value="4")

        self.expect_expr("ClassWithDeletedCtor().value", result_type="int", result_value="6")
        self.expect_expr("ClassWithDeletedDefaultCtor(7).value", result_type="int", result_value="7")

        # FIXME: It seems we try to call the non-existent default constructor here which is wrong.
        self.expect("expr ClassWithDefaultedCtor().foo()", error=True, substrs=["Couldn't lookup symbols:"])

        # FIXME: Calling deleted constructors should fail before linking.
        self.expect("expr ClassWithDeletedCtor(1).value", error=True, substrs=["Couldn't lookup symbols:"])
        self.expect("expr ClassWithDeletedDefaultCtor().value", error=True, substrs=["Couldn't lookup symbols:"])

    @skipIfWindows # Can't find operator new.
    @skipIfLinux # Fails on some Linux systems with SIGABRT.
    def test_constructors_new(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,"// break here", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("(new ClassWithOneCtor(1))->value; 1", result_type="int", result_value="1")
