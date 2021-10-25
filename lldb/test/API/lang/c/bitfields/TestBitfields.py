"""Test C bitfields."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def run_to_main(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

    # BitFields exhibit crashes in record layout on Windows
    # (http://llvm.org/pr21800)
    @skipIfWindows
    def test_bits(self):
        self.run_to_main()

        # Check each field of Bits.
        bits_children = [
            ValueCheck(type="int:1"), # Unnamed and uninitialized
            ValueCheck(type="uint32_t:1", name="b1", value="1"),
            ValueCheck(type="uint32_t:2", name="b2", value="3"),
            ValueCheck(type="int:2"), # Unnamed and uninitialized
            ValueCheck(type="uint32_t:3", name="b3", value="7"),
            ValueCheck(type="uint32_t", name="b4", value="15"),
            ValueCheck(type="uint32_t:5", name="b5", value="31"),
            ValueCheck(type="uint32_t:6", name="b6", value="63"),
            ValueCheck(type="uint32_t:7", name="b7", value="127"),
            ValueCheck(type="uint32_t:4", name="four", value="15")
        ]
        self.expect_var_path("bits", type="Bits", children=bits_children)
        self.expect_expr("bits", result_children=bits_children)

        # Try accessing the different fields using the expression evaluator.
        self.expect_expr("bits.b1", result_type="uint32_t", result_value="1")
        self.expect_expr("bits.b2", result_type="uint32_t", result_value="3")
        self.expect_expr("bits.b3", result_type="uint32_t", result_value="7")
        self.expect_expr("bits.b4", result_type="uint32_t", result_value="15")
        self.expect_expr("bits.b5", result_type="uint32_t", result_value="31")
        self.expect_expr("bits.b6", result_type="uint32_t", result_value="63")
        self.expect_expr("bits.b7", result_type="uint32_t", result_value="127")
        self.expect_expr("bits.four", result_type="uint32_t", result_value="15")

        # Try accessing the different fields using variable paths.
        self.expect_var_path("bits.b1", type="uint32_t:1", value="1")
        self.expect_var_path("bits.b2", type="uint32_t:2", value="3")
        self.expect_var_path("bits.b4", type="uint32_t", value="15")
        self.expect_var_path("bits.b5", type="uint32_t:5", value="31")
        self.expect_var_path("bits.b7", type="uint32_t:7", value="127")


        # Check each field of MoreBits.
        more_bits_children = [
            ValueCheck(type="uint32_t:3", name="a", value="3"),
            ValueCheck(type="int:1", value="0"),
            ValueCheck(type="uint8_t:1", name="b", value="'\\0'"),
            ValueCheck(type="uint8_t:1", name="c", value="'\\x01'"),
            ValueCheck(type="uint8_t:1", name="d", value="'\\0'"),
        ]
        self.expect_var_path("more_bits", type="MoreBits", children=more_bits_children)
        self.expect_expr("more_bits", result_children=more_bits_children)

        self.expect_expr("more_bits.a", result_type="uint32_t", result_value="3")
        self.expect_expr("more_bits.b", result_type="uint8_t", result_value="'\\0'")
        self.expect_expr("more_bits.c", result_type="uint8_t", result_value="'\\x01'")
        self.expect_expr("more_bits.d", result_type="uint8_t", result_value="'\\0'")

        # Test a struct with several single bit fields.
        many_single_bits_children = [
            ValueCheck(type="uint16_t:1", name="b1", value="1"),
            ValueCheck(type="uint16_t:1", name="b2", value="0"),
            ValueCheck(type="uint16_t:1", name="b3", value="0"),
            ValueCheck(type="uint16_t:1", name="b4", value="0"),
            ValueCheck(type="uint16_t:1", name="b5", value="1"),
            ValueCheck(type="uint16_t:1", name="b6", value="0"),
            ValueCheck(type="uint16_t:1", name="b7", value="1"),
            ValueCheck(type="uint16_t:1", name="b8", value="0"),
            ValueCheck(type="uint16_t:1", name="b9", value="0"),
            ValueCheck(type="uint16_t:1", name="b10", value="0"),
            ValueCheck(type="uint16_t:1", name="b11", value="0"),
            ValueCheck(type="uint16_t:1", name="b12", value="0"),
            ValueCheck(type="uint16_t:1", name="b13", value="1"),
            ValueCheck(type="uint16_t:1", name="b14", value="0"),
            ValueCheck(type="uint16_t:1", name="b15", value="0"),
            ValueCheck(type="uint16_t:1", name="b16", value="0"),
            ValueCheck(type="uint16_t:1", name="b17", value="0"),
        ]
        self.expect_var_path("many_single_bits", type="ManySingleBits", children=many_single_bits_children)
        self.expect_expr("many_single_bits", result_type="ManySingleBits",
            result_children=many_single_bits_children)

        # Check a packed struct.
        self.expect_expr("packed.a", result_type="char", result_value="'a'")
        self.expect_expr("packed.b", result_type="uint32_t", result_value="10")
        self.expect_expr("packed.c", result_type="uint32_t", result_value=str(int("7112233", 16)))

        # A packed struct with bitfield size > 32.
        self.expect("v/x large_packed", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["a = 0x0000000cbbbbaaaa", "b = 0x0000000dffffeee"])

        # Check reading a bitfield through a pointer in various ways (PR47743)
        self.expect("v/x large_packed_ptr->b",
                substrs=["large_packed_ptr->b = 0x0000000dffffeeee"])
        self.expect("v/x large_packed_ptr[0].b",
                substrs=["large_packed_ptr[0].b = 0x0000000dffffeeee"])

    # BitFields exhibit crashes in record layout on Windows
    # (http://llvm.org/pr21800)
    @skipIfWindows
    @expectedFailureAll(bugnumber="rdar://74106759")
    def test_expression_bug(self):
        # Ensure evaluating (emulating) an expression does not break bitfield
        # values for already parsed variables. The expression is run twice
        # because the very first expression can resume a target (to allocate
        # memory, etc.) even if it is not being jitted.
        self.run_to_main()

        self.expect("v/x large_packed", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["a = 0x0000000cbbbbaaaa", "b = 0x0000000dffffeee"])
        self.expect("expr --allow-jit false  -- more_bits.a", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '3'])
        self.expect("v/x large_packed", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["a = 0x0000000cbbbbaaaa", "b = 0x0000000dffffeee"])
        self.expect("expr --allow-jit false  -- more_bits.a", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '3'])
        self.expect("v/x large_packed", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["a = 0x0000000cbbbbaaaa", "b = 0x0000000dffffeee"])

    @add_test_categories(['pyapi'])
    # BitFields exhibit crashes in record layout on Windows
    # (http://llvm.org/pr21800)
    @skipIfWindows
    def test_and_python_api(self):
        """Use Python APIs to inspect a bitfields variable."""
        self.run_to_main()

        # Lookup the "bits" variable which contains 8 bitfields.
        bits = self.frame().FindVariable("bits")
        self.DebugSBValue(bits)
        self.assertEqual(bits.GetTypeName(), 'Bits')
        self.assertEqual(bits.GetNumChildren(), 10)
        self.assertEqual(bits.GetByteSize(), 32)

        # Notice the pattern of int(b1.GetValue(), 0).  We pass a base of 0
        # so that the proper radix is determined based on the contents of the
        # string.
        b1 = bits.GetChildMemberWithName("b1")
        self.DebugSBValue(b1)
        self.assertEqual(b1.GetName(), "b1")
        self.assertEqual(b1.GetTypeName(), "uint32_t:1")
        self.assertTrue(b1.IsInScope())
        self.assertEqual(int(b1.GetValue(), 0), 1)

        b7 = bits.GetChildMemberWithName("b7")
        self.assertEqual(b7.GetName(), "b7")
        self.assertEqual(b7.GetTypeName(), "uint32_t:7")
        self.assertTrue(b7.IsInScope())
        self.assertEqual(int(b7.GetValue(), 0), 127)

        four = bits.GetChildMemberWithName("four")
        self.assertEqual(four.GetName(), "four")
        self.assertEqual(four.GetTypeName(), "uint32_t:4")
        self.assertTrue(four.IsInScope())
        self.assertEqual(int(four.GetValue(), 0), 15)