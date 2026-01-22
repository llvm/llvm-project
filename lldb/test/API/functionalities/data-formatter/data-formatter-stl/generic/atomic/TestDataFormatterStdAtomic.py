"""
Test lldb data formatter subsystem.
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdAtomicTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def get_variable(self, name):
        var = self.frame().FindVariable(name)
        var.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        return var

    def do_test(self, test_smart_pointers: bool = False):
        """Test that std::atomic is correctly printed by LLDB"""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."
            )
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        s_atomic = self.get_variable("s")
        i_atomic = self.get_variable("i")

        if self.TraceOn():
            print(s_atomic)
        if self.TraceOn():
            print(i_atomic)

        # Extract the content of the std::atomic wrappers.
        self.assertEqual(s_atomic.GetNumChildren(), 1)
        s = s_atomic.GetChildAtIndex(0)
        self.assertEqual(i_atomic.GetNumChildren(), 1)
        i = i_atomic.GetChildAtIndex(0)

        self.assertEqual(i.GetValueAsUnsigned(0), 5, "i == 5")
        self.assertEqual(s.GetNumChildren(), 2, "s has two children")
        self.assertEqual(s.GetChildAtIndex(0).GetValueAsUnsigned(0), 1, "s.x == 1")
        self.assertEqual(s.GetChildAtIndex(1).GetValueAsUnsigned(0), 2, "s.y == 2")

        # float
        atomic_float = self.get_variable("atomic_float")
        val_float = atomic_float.child[0]
        self.assertIsNotNone(val_float, msg="atomic_float child is None.")
        self.assertAlmostEqual(float(val_float.value), 3.14, places=2)
        # double
        atomic_double = self.get_variable("atomic_double")
        val_float = atomic_double.child[0]
        self.assertIsNotNone(val_float, msg="atomic_double child is None.")
        self.assertAlmostEqual(float(val_float.value), 6.28, places=2)

        # bool
        atomic_bool = self.get_variable("atomic_bool")
        val_bool = atomic_bool.child[0]
        self.assertIsNotNone(val_bool, msg="atomic_bool child is None.")
        self.assertEqual(bool(val_bool.unsigned), True)

        # function
        atomic_func = self.get_variable("atomic_func")
        val_func = atomic_func.child[0]
        self.assertIsNotNone(val_func, msg="atomic_func child is None.")
        self.assertNotEqual(val_func.unsigned, 0)

        # pointer
        atomic_pointer = self.get_variable("atomic_pointer")
        val_pointer = atomic_pointer.child[0]
        self.assertIsNotNone(val_pointer, msg="atomic_pointer child is None.")
        self.assertNotEqual(val_pointer.unsigned, 0)

        # Try printing the child that points to its own parent object.
        # This should just treat the atomic pointer as a normal pointer.
        self.expect("frame var p.child", substrs=["Value = 0x"])
        self.expect("frame var p", substrs=["parent = {", "Value = 0x", "}"])
        self.expect(
            "frame var p.child.parent", substrs=["p.child.parent = {\n  Value = 0x"]
        )

        if test_smart_pointers:
            # shared_pointer
            atomic_shared = self.get_variable("atomic_shared")
            val_shared = atomic_shared.child[0]
            self.assertIsNotNone(val_shared, msg="atomic_shared child is None.")
            self.assertEqual(300, val_shared.deref.unsigned)

            # weak_pointer
            atomic_weak = self.get_variable("atomic_weak")
            val_weak = atomic_weak.child[0]
            self.assertIsNotNone(val_weak, msg="atomic_weak child is None.")
            self.assertEqual(300, val_weak.deref.unsigned)

    def verify_floating_point_equal(self, value: str, expected: float):
        self.assertAlmostEqual(float(value), float(expected), places=4)

    @skipIf(compiler=["gcc"])
    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    # the data layout is different in new libstdc++ versions
    @add_test_categories(["libstdcxx"])
    def test_libstdcxx_11(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1, "CXXFLAGS_EXTRAS": "-std=c++11"})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx_17(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1, "CXXFLAGS_EXTRAS": "-std=c++17"})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx_20(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1, "CXXFLAGS_EXTRAS": "-std=c++20"})
        self.do_test(test_smart_pointers=True)

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test()
