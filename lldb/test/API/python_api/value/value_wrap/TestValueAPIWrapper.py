"""
Test the lldb.value wrapper.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueAPIWrapper(TestBase):
    def test_accessors(self):
        """Test non-modifying operators (e.g. __getitem__, __add__)."""
        self.build()

        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)

        # Get the variables and check __bool__.
        u32_zero = lldb.value(frame.FindVariable("u32_zero"))
        self.assertTrue(u32_zero)
        u32_one = lldb.value(frame.FindVariable("u32_one"))
        self.assertTrue(u32_one)
        u32_two = lldb.value(frame.FindVariable("u32_two"))
        self.assertTrue(u32_two)
        u32_four = lldb.value(frame.FindVariable("u32_four"))
        self.assertTrue(u32_four)

        i32_zero = lldb.value(frame.FindVariable("i32_zero"))
        self.assertTrue(i32_zero)
        i32_one = lldb.value(frame.FindVariable("i32_one"))
        self.assertTrue(i32_one)
        i32_two = lldb.value(frame.FindVariable("i32_two"))
        self.assertTrue(i32_two)

        i32_minus_one = lldb.value(frame.FindVariable("i32_minus_one"))
        self.assertTrue(i32_minus_one)
        i32_minus_two = lldb.value(frame.FindVariable("i32_minus_two"))
        self.assertTrue(i32_minus_two)

        cstr = lldb.value(frame.FindVariable("cstr"))
        self.assertTrue(cstr)

        arr = lldb.value(frame.FindVariable("arr"))
        self.assertTrue(arr)
        arr_start = lldb.value(frame.FindVariable("arr_start"))
        self.assertTrue(arr_start)
        arr_second = lldb.value(frame.FindVariable("arr_second"))
        self.assertTrue(arr_second)

        my_car = lldb.value(frame.FindVariable("my_car"))
        self.assertTrue(my_car)

        self.assertFalse(lldb.value(frame.FindVariable("this_does_not_exist")))

        # Test __str__().
        self.assertEqual(str(u32_zero), "(uint32_t) u32_zero = 0")
        self.assertEqual(str(u32_one), "(uint32_t) u32_one = 1")

        # Test __getitem__(key).
        self.assertIsInstance(arr[0], lldb.value)
        self.assertEqual(int(arr[0]), 1)
        self.assertEqual(int(arr[1]), 2)
        self.assertEqual(int(arr[u32_two]), 3)
        self.assertEqual(int(arr[i32_two]), 3)
        with self.assertRaisesRegex(IndexError, "^Index '1' is out of range$"):
            my_car[1]
        with self.assertRaisesRegex(TypeError, "^No array item of type <class 'str'>$"):
            my_car["engine"]

        # Test __iter__():
        self.assertTrue(all(isinstance(v, lldb.value) for v in arr))
        self.assertEqual(list(int(v) for v in arr), [1, 2, 3, 4, 5, 6])

        # Test __getattr__():
        self.assertIsInstance(my_car.doors, lldb.value)
        self.assertEqual(int(my_car.doors), 3)
        self.assertEqual(int(my_car.wheels), 4)
        self.assertEqual(int(my_car.engine.kind), 1)
        with self.assertRaisesRegex(
            AttributeError, "^Attribute 'windows' is not defined$"
        ):
            my_car.windows

        # Test __add__(other).
        self.assertEqual(u32_one + i32_one, 2)
        self.assertEqual(u32_one + 1, 2)
        self.assertEqual(u32_one + i32_minus_one, 0)
        self.assertEqual(i32_minus_two + u32_one, -1)
        # Pointers use byte addresses (u16 *arr_start).
        self.assertEqual(arr_start + 2, arr[1].sbvalue.GetLoadAddress())

        # Test __sub__(other).
        self.assertEqual(u32_two - 1, 1)
        self.assertEqual(i32_minus_one - 4, -5)
        self.assertEqual(u32_one - 4, -3)
        self.assertEqual(arr_second - 2, int(arr_start))

        # Test __mul__(other).
        self.assertEqual(u32_two * 3, 6)
        self.assertEqual(i32_one * 5, 5)
        self.assertEqual(i32_minus_one * 3, -3)
        self.assertEqual(i32_minus_one * i32_minus_two, 2)

        # Test __floordiv__(other).
        self.assertEqual(u32_two // 2, 1)
        self.assertEqual(i32_minus_two // i32_one, -2)

        # Test __mod__(other).
        self.assertEqual(u32_two % 2, 0)
        self.assertEqual(i32_minus_two % u32_two, 0)

        # Test __divmod__(other).
        # FIXME: Returns one number right now - should return a tuple.
        # self.assertEqual(divmod(u32_four, 3), divmod(4, 3))
        # self.assertEqual(divmod(u32_four, i32_two), divmod(4, 2))

        # Test __pow__(other).
        self.assertEqual(u32_two**2, 4)
        self.assertEqual(i32_two**u32_four, 16)

        # Test __lshift__(other).
        self.assertEqual(u32_one << 2, 4)
        self.assertEqual(i32_two << u32_four, 32)

        # Test __rshift__(other).
        self.assertEqual(i32_two >> 1, 1)
        self.assertEqual(u32_four >> i32_one, 2)

        # Test __and__(other).
        self.assertEqual(my_car.doors & 2, 2)
        self.assertEqual(my_car.doors & u32_one, 1)

        # Test __xor__(other).
        self.assertEqual(my_car.doors ^ 0b101, 0b110)
        self.assertEqual(my_car.doors ^ u32_one, 0b10)

        # Test __or__(other).
        self.assertEqual(u32_one | 2, 3)
        self.assertEqual(u32_four | u32_two, 6)

        # Test __truediv__(other).
        self.assertEqual(u32_two / 2, 1.0)
        self.assertEqual(my_car.doors / u32_two, 3 / 2)

        # Test __neg__().
        self.assertEqual(-i32_one, -1)
        self.assertEqual(-i32_minus_two, 2)

        # Test __pos__().
        self.assertEqual(+i32_one, 1)
        self.assertEqual(+i32_minus_two, -2)

        # Test __abs__().
        self.assertEqual(abs(i32_minus_one), 1)
        self.assertEqual(abs(u32_two), 2)

        # Test __invert__().
        self.assertEqual(~i32_one, -2)

        # Test __complex__().
        self.assertEqual(complex(u32_one), complex(1, 0))

        # Test __len__().
        self.assertEqual(len(u32_four), 0)
        self.assertEqual(len(arr), 6)
        self.assertEqual(len(my_car), 3)

        # Test __eq__().
        self.assertEqual(u32_two, i32_two)
        self.assertEqual(u32_two, 2)
        self.assertEqual(u32_two, "(uint32_t) u32_two = 2")
        with self.assertRaisesRegex(
            TypeError, "^Unknown type <class 'bool'>, No equality operation defined.$"
        ):
            _unused = u32_one == True

        # Test __ne__().
        self.assertNotEqual(u32_two, 1)
        self.assertNotEqual(u32_two, i32_one)
        self.assertNotEqual(u32_two, "2")
        with self.assertRaisesRegex(
            TypeError, "^Unknown type <class 'bool'>, No equality operation defined.$"
        ):
            _unused = u32_one != True

        # FIXME: Missing __index__ for oct(), hex(), etc.

    def test_in_place_modifiers(self):
        """Test in-place operators (__i...__(self, other))."""
        self.build()

        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)

        engine = lldb.value(frame.FindVariable("engine"))
        self.assertTrue(engine)

        kind = engine.kind
        self.assertEqual(kind, 1)
        self.assertIsInstance(kind, lldb.value)

        # Test __iadd__(other).
        kind += 1
        self.assertEqual(kind, 2)
        self.assertIsInstance(kind, int)
        kind = engine.kind
        self.assertEqual(kind, 2)

        # Test __isub__(other).
        kind -= 1
        self.assertEqual(kind, 1)
        self.assertIsInstance(kind, int)
        kind = engine.kind
        self.assertEqual(kind, 1)

        # Test __imul__(other).
        kind *= 6
        self.assertEqual(kind, 6)
        self.assertIsInstance(kind, int)
        kind = engine.kind
        self.assertEqual(kind, 6)

        # Test __itruediv__(other).
        kind /= 2
        self.assertEqual(kind, 3)
        self.assertIsInstance(kind, float)
        kind = engine.kind
        # Keeps its value, because we try to set "3.0".
        # FIXME: Raise error here.
        self.assertEqual(kind, 6)

        # Test __ifloordiv__(other).
        # FIXME: Passes too many arguments to __floordiv__.

        # Test __imod__(other).
        # FIXME: Passes too many arguments to __mod__.

        # Test __ipow__(other).
        # FIXME: Passes too many arguments to __pow__.

        # Reset value
        kind *= 0
        kind = engine.kind
        kind += 1
        kind = engine.kind
        self.assertEqual(kind, 1)
        self.assertIsInstance(kind, lldb.value)

        # Test __ilshift__(other).
        kind <<= 3
        self.assertEqual(kind, 8)
        self.assertIsInstance(kind, int)
        kind = engine.kind
        self.assertEqual(kind, 8)

        # Test __irshift__(other).
        kind >>= 1
        self.assertEqual(kind, 4)
        self.assertIsInstance(kind, int)
        kind = engine.kind
        self.assertEqual(kind, 4)

        # Test __iand__(other).
        # FIXME: Passes too many arguments to __and__.

        # Test __ixor__(other).
        # FIXME: Passes too many arguments to __xor__.

        # Test __ior__(other).
        # FIXME: Passes too many arguments to __or__.

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 4)  # Last value of `engine.kind`.
