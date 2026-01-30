import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.selected_frame
        array_type = frame.var("array").type
        generic_type = array_type.GetTemplateArgumentType(0)
        self.assertTrue(generic_type.IsValid())
        self.assertEqual(generic_type.name, "Swift.Int")
        self.assertEqual(generic_type.GetDisplayTypeName(), "Int")
