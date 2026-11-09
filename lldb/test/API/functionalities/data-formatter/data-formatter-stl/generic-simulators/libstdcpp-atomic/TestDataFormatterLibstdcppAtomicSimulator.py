"""
Test we can understand the layouts of libstdc++'s std::atomic that change
between libstdc++ versions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibstdcppAtomicSimulatorTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    def get_inner(self, name):
        """Returns the value wrapped in the std::atomic `name`."""
        atomic = self.frame().FindVariable(name)
        self.assertTrue(atomic.IsValid(), msg=f"{name} is not valid.")
        atomic.SetPreferSyntheticValue(True)
        self.assertEqual(atomic.GetNumChildren(), 1, msg=f"{name} has one child.")
        inner = atomic.GetChildAtIndex(0)
        self.assertTrue(inner.IsValid(), msg=f"{name} child is not valid.")
        self.assertEqual(inner.GetName(), "Value")
        return inner

    def do_test(self, float_layout: int, refcount_layout: int):
        """
        float_layout:
            - 0 uses the generic `_M_i` for std::atomic<float>.
            - 1 uses the specialization with `_M_fp`.
        refcount_layout:
            - 0 uses a scalar `_M_val` for the reference count of std::atomic<std::shared_ptr<T>>.
            - 1 wraps it in a struct with `_M_i`.
        """
        defines = [
            f"-DFLOAT_LAYOUT={float_layout}",
            f"-DREFCOUNT_LAYOUT={refcount_layout}",
        ]
        cxx_flags = " ".join(defines)
        self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxx_flags))
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.assertEqual(self.get_inner("atomic_int").GetValueAsSigned(), 5)

        atomic_struct = self.get_inner("atomic_struct")
        self.assertEqual(atomic_struct.GetNumChildren(), 2)
        self.assertEqual(atomic_struct.GetChildAtIndex(0).GetValueAsSigned(), 1)
        self.assertEqual(atomic_struct.GetChildAtIndex(1).GetValueAsSigned(), 2)

        self.assertEqual(self.get_inner("atomic_bool").GetValueAsUnsigned(), 1)

        # The value is in `_M_fp` or `_M_i` depending on the layout.
        self.assertAlmostEqual(float(self.get_inner("atomic_float").value), 3.14, 2)
        self.assertAlmostEqual(float(self.get_inner("atomic_double").value), 6.28, 2)

        # Complex and vector floating point types use the generic layout.
        self.get_inner("atomic_complex_float")
        self.assertEqual(self.get_inner("atomic_vector_float").GetNumChildren(), 4)

        atomic_pointer = self.get_inner("atomic_pointer")
        self.assertEqual(atomic_pointer.Dereference().GetValueAsSigned(), 42)

        # The summary of atomic smart pointers reads the reference count.
        atomic_shared = self.get_inner("atomic_shared")
        self.assertEqual(atomic_shared.Dereference().GetValueAsSigned(), 300)
        self.expect("frame variable atomic_shared", substrs=["strong=1", "weak=0"])

        atomic_weak = self.get_inner("atomic_weak")
        self.assertEqual(atomic_weak.Dereference().GetValueAsSigned(), 300)
        self.expect("frame variable atomic_weak", substrs=["strong=1", "weak=1"])

    def test_generic_float_scalar_refcount(self):
        # std::atomic<float> uses `_M_i` and the reference count is a scalar.
        self.do_test(float_layout=0, refcount_layout=0)

    def test_specialized_float_struct_refcount(self):
        # std::atomic<float> uses `_M_fp` and the reference count is a struct.
        self.do_test(float_layout=1, refcount_layout=1)
