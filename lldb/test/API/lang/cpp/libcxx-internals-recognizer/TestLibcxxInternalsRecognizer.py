import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import re

class LibCxxInternalsRecognizerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["libc++"])
    @skipIf(compiler="clang", compiler_version=["<", "19.0"])
    def test_frame_recognizer(self):
        """Test that implementation details of libc++ are hidden"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        expected_parents = {
            "sort_less(int, int)": ["::sort", "test_algorithms"],
            # `std::ranges::sort` is implemented as an object of types `__sort`.
            # We never hide the frame of the entry-point into the standard library, even
            # if the name starts with `__` which usually indicates an internal function.
            "ranges_sort_less(int, int)": [
                re.compile(r"ranges::__sort::(__fn::)?operator\(\)"),
                "test_algorithms",
            ],
            # `ranges::views::transform` internally uses `std::invoke`, and that
            # call also shows up in the stack trace
            "view_transform(int)": [
                "::invoke",
                "ranges::transform_view",
                "test_algorithms",
            ],
            # Various types of `invoke` calls
            "consume_number(int)": ["::invoke", "test_invoke"],
            "invoke_add(int, int)": ["::invoke", "test_invoke"],
            "Callable::member_function(int) const": ["::invoke", "test_invoke"],
            "Callable::operator()(int) const": ["::invoke", "test_invoke"],
            # Containers
            "MyKey::operator<(MyKey const&) const": [
                "::operator()",
                "less",
                "::emplace",
                "test_containers",
            ],
        }
        stop_set = set()
        while process.GetState() != lldb.eStateExited:
            fn = thread.GetFrameAtIndex(0).GetFunctionName()
            stop_set.add(fn)
            self.assertIn(fn, expected_parents.keys())
            frame_id = 1
            for expected_parent in expected_parents[fn]:
                # Skip all hidden frames
                while (
                    frame_id < thread.GetNumFrames()
                    and thread.GetFrameAtIndex(frame_id).IsHidden()
                ):
                    frame_id = frame_id + 1
                # Expect the correct parent frame
                func_name = thread.GetFrameAtIndex(frame_id).GetFunctionName()
                if isinstance(expected_parent, re.Pattern):
                    self.assertTrue(
                        expected_parent.search(func_name) is not None,
                        f"'{expected_parent}' not found in '{func_name}'"
                    )
                else:
                    self.assertIn(expected_parent, func_name)
                frame_id = frame_id + 1
            process.Continue()

        # Make sure that we actually verified all intended scenarios
        self.assertEqual(len(stop_set), len(expected_parents))
