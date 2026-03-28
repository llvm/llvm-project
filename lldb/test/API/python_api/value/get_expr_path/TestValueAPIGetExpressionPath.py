import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueAPIGetExpressionPath(TestBase):
    def test(self):
        self.build()

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Break at this line", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)

        self.assertEqual(frame.FindVariable("foo").get_expr_path(), "foo")
        for i in range(2):
            self.assertEqual(
                frame.FindVariable("foo").GetChildAtIndex(i).get_expr_path(),
                f"foo[{i}]",
            )
            for j in range(3):
                self.assertEqual(
                    frame.FindVariable("foo")
                    .GetChildAtIndex(i)
                    .GetChildAtIndex(j)
                    .get_expr_path(),
                    f"foo[{i}][{j}]",
                )
                for k in range(4):
                    self.assertEqual(
                        frame.FindVariable("foo")
                        .GetChildAtIndex(i)
                        .GetChildAtIndex(j)
                        .GetChildAtIndex(k)
                        .get_expr_path(),
                        f"foo[{i}][{j}][{k}]",
                    )
        self.assertEqual(frame.FindVariable("bar").get_expr_path(), "bar")
        for j in range(3):
            self.assertEqual(
                frame.FindVariable("bar").GetChildAtIndex(j).get_expr_path(),
                f"bar[0][{j}]",
            )
            for k in range(4):
                self.assertEqual(
                    frame.FindVariable("bar")
                    .GetChildAtIndex(j)
                    .GetChildAtIndex(k)
                    .get_expr_path(),
                    f"bar[0][{j}][{k}]",
                )
