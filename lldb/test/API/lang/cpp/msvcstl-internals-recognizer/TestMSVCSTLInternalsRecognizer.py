import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def _qualified_name_in_std(name):
    """Return True if `name` is a function in the `std::` namespace.

    Handles the MSVC demangler convention of prefixing the demangled name
    with the return type (e.g. `void std::_Func_class<void>::operator()`).
    """
    if not name:
        return False
    return name.startswith("std::") or " std::" in name


class MSVCSTLInternalsRecognizerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _run_to_target(self):
        self.build()
        return lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

    @skipUnlessWindows
    def test_frame_recognizer(self):
        """At least one MSVC STL internal frame between `target` and `main`
        should be hidden, but not all frames."""
        target, process, thread, bkpt = self._run_to_target()

        self.assertIn("target", thread.GetFrameAtIndex(0).GetFunctionName())

        num_hidden = sum(1 for frame in thread.frames if frame.IsHidden())
        self.assertGreater(num_hidden, 0)
        self.assertLess(num_hidden, thread.GetNumFrames())

    @skipUnlessWindows
    def test_outermost_std_frame_visible(self):
        """The outermost `std::*` frame (called directly by user code) must
        stay visible."""
        target, process, thread, bkpt = self._run_to_target()

        for i in range(thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if not frame.IsHidden():
                continue
            name = frame.GetFunctionName() or ""
            self.assertTrue(
                _qualified_name_in_std(name),
                f"hidden frame #{i} '{name}' should be in 'std::' namespace",
            )
            parent = thread.GetFrameAtIndex(i + 1)
            self.assertIsNotNone(parent)
            parent_name = parent.GetFunctionName() or ""
            self.assertTrue(
                _qualified_name_in_std(parent_name),
                f"hidden frame #{i} '{name}' has non-std parent '{parent_name}'",
            )

    @skipUnlessWindows
    def test_backtrace(self):
        """`bt` hides MSVC STL internals; `bt -u` shows them."""
        target, process, thread, bkpt = self._run_to_target()

        self.expect(
            "thread backtrace",
            ordered=True,
            substrs=["frame", "target", "frame", "main"],
        )
        self.expect(
            "thread backtrace",
            matching=False,
            patterns=[r"frame.*std::_Func_impl", r"frame.*_Do_call"],
        )
        self.expect(
            "thread backtrace -u",
            ordered=True,
            patterns=[r"frame.*target", r"frame.*std::_", r"frame.*main"],
        )
        self.expect(
            "bt -u",
            ordered=True,
            patterns=[r"frame.*target", r"frame.*std::_", r"frame.*main"],
        )
        self.expect(
            "thread backtrace --unfiltered",
            ordered=True,
            patterns=[r"frame.*target", r"frame.*std::_", r"frame.*main"],
        )

    @skipUnlessWindows
    def test_up_down(self):
        """`up` and `down` should skip past hidden MSVC STL frames."""
        target, process, thread, bkpt = self._run_to_target()

        frame = thread.selected_frame
        self.assertIn("target", frame.GetFunctionName())
        start_idx = frame.GetFrameID()

        # Walk up until we hit `main`. The number of `up` invocations must be
        # less than the raw frame distance, proving hidden frames were skipped.
        up_steps = 0
        for _ in range(thread.GetNumFrames()):
            self.expect("up")
            up_steps += 1
            frame = thread.selected_frame
            if frame.GetFunctionName() == "main":
                break
        end_idx = frame.GetFrameID()
        self.assertEqual(frame.GetFunctionName(), "main")
        self.assertLess(
            up_steps, end_idx - start_idx, "expected skipped frames going up"
        )

        # Walk back down to `target`.
        start_idx = frame.GetFrameID()
        down_steps = 0
        for _ in range(thread.GetNumFrames()):
            self.expect("down")
            down_steps += 1
            frame = thread.selected_frame
            if "target" in (frame.GetFunctionName() or ""):
                break
        end_idx = frame.GetFrameID()
        self.assertIn("target", frame.GetFunctionName())
        self.assertLess(
            down_steps, start_idx - end_idx, "expected skipped frames going down"
        )

    @skipUnlessWindows
    def test_user_lambda_not_hidden(self):
        """The user's lambda wrapper (`main::<lambda_...>::operator()`)
        is user code and must NOT be hidden, even though it sits among
        `std::function` machinery."""
        target, process, thread, bkpt = self._run_to_target()

        for i in range(thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            name = frame.GetFunctionName() or ""
            if "lambda" in name and "::operator()" in name:
                self.assertFalse(
                    frame.IsHidden(),
                    f"user lambda frame '{name}' should not be hidden",
                )
                return
        self.fail("did not find a user lambda frame in the backtrace")
