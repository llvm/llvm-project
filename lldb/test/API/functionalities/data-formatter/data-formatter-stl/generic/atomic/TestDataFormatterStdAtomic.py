"""
Test lldb data formatter subsystem.
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdAtomicTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def get_variable(self, name):
        var = self.frame().FindVariable(name)
        var.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        return var

    def do_test(self, check_shared_ptr: bool = True):
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

        # float, only available if std::atomic<float> is supported.
        atomic_float = self.get_variable("atomic_float")
        if atomic_float.IsValid():
            val_float = atomic_float.child[0]
            self.assertIsNotNone(val_float, msg="atomic_float child is None.")
            self.assertAlmostEqual(float(val_float.value), 3.14, places=2)
            # double
            atomic_double = self.get_variable("atomic_double")
            val_double = atomic_double.child[0]
            self.assertIsNotNone(val_double, msg="atomic_double child is None.")
            self.assertAlmostEqual(float(val_double.value), 6.28, places=2)

        # complex float, only available with libstdc++.
        atomic_complex_float = self.get_variable("atomic_complex_float")
        if atomic_complex_float.IsValid():
            self.assertEqual(atomic_complex_float.GetNumChildren(), 1)
            val_complex = atomic_complex_float.child[0]
            self.assertIsNotNone(val_complex, msg="atomic_complex_float child is None.")
            self.assertTrue(val_complex.IsValid())

        # vector of floats
        atomic_vector_float = self.get_variable("atomic_vector_float")
        self.assertEqual(atomic_vector_float.GetNumChildren(), 1)
        val_vector = atomic_vector_float.child[0]
        self.assertIsNotNone(val_vector, msg="atomic_vector_float child is None.")
        self.assertEqual(val_vector.GetNumChildren(), 4)

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

        # Atomic smart pointers, only available if supported by the library.
        atomic_shared = self.get_variable("atomic_shared")
        if check_shared_ptr and atomic_shared.IsValid():
            # shared_pointer
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

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test(check_shared_ptr=False)
