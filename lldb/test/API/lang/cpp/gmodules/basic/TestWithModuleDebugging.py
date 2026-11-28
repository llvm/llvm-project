import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWithGmodulesDebugInfo(TestBase):
    SHARED_BUILD_TESTCASE = False

    @skipIf(bugnumber="llvm.org/pr36146", oslist=["linux"], archs=["i386"])
    @add_test_categories(["gmodules"])
    def test_specialized_typedef_from_pch(self):
        self.build()

        src_file = os.path.join(self.getSourceDir(), "main.cpp")
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "breakpoint file")

        # Break on interesting line
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", src_file_spec
        )

        # Get frame for current thread
        frame = thread.frames[0]

        testValue = frame.EvaluateExpression("test")
        self.assertTrue(
            testValue.GetError().Success(),
            "Test expression value invalid: %s" % (testValue.GetError().GetCString()),
        )
        self.assertEqual(
            testValue.GetTypeName(), "IntContainer", "Test expression type incorrect"
        )

        memberValue = testValue.GetChildMemberWithName("storage")
        self.assertTrue(
            memberValue.GetError().Success(),
            "Member value missing or invalid: %s" % (testValue.GetError().GetCString()),
        )
        self.assertEqual(memberValue.GetTypeName(), "int", "Member type incorrect")
        self.assertEqual(42, memberValue.GetValueAsSigned(), "Member value incorrect")

        testValue = frame.EvaluateExpression("bar")
        self.assertTrue(
            testValue.GetError().Success(),
            "Test expression value invalid: %s" % (testValue.GetError().GetCString()),
        )
        self.assertEqual(
            testValue.GetTypeName(), "Foo::Bar", "Test expression type incorrect"
        )

        memberValue = testValue.GetChildMemberWithName("i")
        self.assertTrue(
            memberValue.GetError().Success(),
            "Member value missing or invalid: %s" % (testValue.GetError().GetCString()),
        )
        self.assertEqual(memberValue.GetTypeName(), "int", "Member type incorrect")
        self.assertEqual(123, memberValue.GetValueAsSigned(), "Member value incorrect")
