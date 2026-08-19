"""Simulate MSVC STL layouts and check the corresponding LLDB formatters."""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def check_adaptor(self, name, values):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid(), name)
        self.assertEqual(var.GetNumChildren(), len(values), name)
        for i, expected in enumerate(values):
            child = var.GetChildAtIndex(i)
            self.assertTrue(child.IsValid(), f"{name}[{i}]")
            self.assertEqual(child.GetValueAsSigned(), expected, f"{name}[{i}]")
            self.assertEqual(child.GetName(), f"[{i}]")

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        empty_bitset = self.frame().FindVariable("empty_bitset")
        self.assertEqual(empty_bitset.GetNumChildren(), 0)
        self.expect("frame variable empty_bitset", substrs=["size=0"])

        small_bitset = self.frame().FindVariable("small_bitset")
        self.assertEqual(small_bitset.GetNumChildren(), 13)
        self.assertEqual(small_bitset.GetChildAtIndex(2).GetValueAsUnsigned(), 1)
        self.assertEqual(small_bitset.GetChildAtIndex(11).GetValueAsUnsigned(), 0)
        self.expect(
            "frame variable small_bitset",
            substrs=["size=13", "[2] = true", "[9] = true", "[11] = false"],
        )

        large_bitset = self.frame().FindVariable("large_bitset")
        self.assertEqual(large_bitset.GetNumChildren(), 70)
        self.assertEqual(large_bitset.GetChildAtIndex(0).GetValueAsUnsigned(), 1)
        self.assertEqual(large_bitset.GetChildAtIndex(1).GetValueAsUnsigned(), 0)
        # `_Array[1] = 1` sets the first bit of the second word: 32 on LLP64,
        # 64 on LP64.
        second_word_bit = None
        for candidate in (32, 64):
            child = large_bitset.GetChildAtIndex(candidate)
            if child.IsValid() and child.GetValueAsUnsigned() == 1:
                second_word_bit = candidate
                break
        self.assertIsNotNone(second_word_bit)

        ili = self.frame().FindVariable("ili")
        self.assertEqual(ili.GetNumChildren(), 5)
        self.assertEqual(ili.GetChildAtIndex(0).GetValueAsSigned(), 1)
        self.assertEqual(ili.GetChildAtIndex(4).GetValueAsSigned(), 5)

        vec = self.frame().FindVariable("vec")
        self.assertEqual(vec.GetNumChildren(), 3)
        self.check_adaptor("q", [10, 20, 30])
        self.check_adaptor("st", [10, 20, 30])
        self.check_adaptor("pq", [10, 20, 30])

        va = self.frame().FindVariable("va")
        self.assertEqual(va.GetNumChildren(), 4)
        self.assertEqual(va.GetChildAtIndex(0).GetValueAsSigned(), 1)
        self.assertEqual(va.GetChildAtIndex(3).GetValueAsSigned(), 1234)
        self.expect("frame variable va", substrs=["size=4"])

        va_empty = self.frame().FindVariable("va_empty")
        self.assertEqual(va_empty.GetNumChildren(), 0)
        self.expect("frame variable va_empty", substrs=["size=0"])

        va_ref = self.frame().FindVariable("va_ref")
        self.assertEqual(va_ref.GetNumChildren(), 4)
        self.assertEqual(va_ref.GetChildAtIndex(3).GetValueAsSigned(), 1234)
        self.expect("frame variable va_ref", substrs=["size=4", "[0] = 1"])

        ok = self.frame().FindVariable("ok")
        self.assertEqual(ok.GetNumChildren(), 1)
        self.assertEqual(ok.GetChildAtIndex(0).GetName(), "Value")
        self.assertEqual(ok.GetChildAtIndex(0).GetValueAsSigned(), 7)
        self.assertFalse(ok.GetChildMemberWithName("Unexpected").IsValid())
        self.expect("frame variable ok", substrs=["Has Value=true", "Value = 7"])

        err = self.frame().FindVariable("err")
        self.assertEqual(err.GetNumChildren(), 1)
        self.assertEqual(err.GetChildAtIndex(0).GetName(), "Unexpected")
        self.assertFalse(err.GetChildMemberWithName("Value").IsValid())
        self.expect(
            "frame variable err",
            substrs=["Has Value=false", "Unexpected =", "boom"],
        )

        void_ok = self.frame().FindVariable("void_ok")
        self.assertEqual(void_ok.GetNumChildren(), 0)
        self.expect("frame variable void_ok", substrs=["Has Value=true"])

        void_err = self.frame().FindVariable("void_err")
        self.assertEqual(void_err.GetNumChildren(), 1)
        self.assertEqual(void_err.GetChildAtIndex(0).GetName(), "Unexpected")
        self.assertEqual(void_err.GetChildAtIndex(0).GetValueAsSigned(), 11)
        self.expect("frame variable void_err", substrs=["Has Value=false"])

        self.expect(
            "frame variable loc",
            substrs=['"main.cpp":6:1', "__cdecl", "main"],
        )
        loc_empty = self.frame().FindVariable("loc_empty")
        self.assertTrue(loc_empty.GetError().Success())
        self.assertTrue(not loc_empty.summary)

        self.expect("frame variable ns", substrs=["ns = 1 ns"])
        self.expect("frame variable s", substrs=["s = 1234 s"])
        self.expect("frame variable custom_dur", substrs=["custom_dur = 42"])

        ec = self.frame().FindVariable("ec")
        self.assertGreaterEqual(ec.GetNumChildren(), 1)
        self.expect("frame variable ec", substrs=["value=2"])
        econd = self.frame().FindVariable("econd")
        self.assertGreaterEqual(econd.GetNumChildren(), 1)
        self.expect("frame variable econd", substrs=["value=7"])

        self.expect("frame variable p", substrs=["file.txt"])

        it = self.frame().FindVariable("it")
        self.assertEqual(it.GetNumChildren(), 1)
        self.assertEqual(it.GetChildAtIndex(0).GetName(), "item")
        self.assertEqual(it.GetChildAtIndex(0).GetValueAsSigned(), 3)
        cit = self.frame().FindVariable("cit")
        self.assertEqual(cit.GetChildAtIndex(0).GetValueAsSigned(), 3)
