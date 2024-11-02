"""
Tests the builtin formats of LLDB.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    def getFormatted(self, format, expr):
        """
        Evaluates the expression and formats the result with the given format.
        """
        result = lldb.SBCommandReturnObject()
        full_expr = "expr --format '" + format + "' -- " + expr
        self.dbg.GetCommandInterpreter().HandleCommand(full_expr, result)
        self.assertTrue(result.Succeeded(), result.GetError())
        return result.GetOutput()

    @no_debug_info_test
    @skipIfWindows
    # uint128_t not available on arm.
    @skipIf(archs=['arm'])
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # void
        self.assertEqual("", self.getFormatted("void", "1"))

        # boolean
        self.assertIn("= false\n", self.getFormatted("boolean", "0"))
        self.assertIn("= true\n", self.getFormatted("boolean", "1"))
        self.assertIn("= true\n", self.getFormatted("boolean", "2"))
        self.assertIn("= error: unsupported byte size (16) for boolean format\n", self.getFormatted("boolean", "(__uint128_t)0"))

        # float
        self.assertIn("= 0\n", self.getFormatted("float", "0"))
        self.assertIn("= 2\n", self.getFormatted("float", "0x40000000"))
        self.assertIn("= NaN\n", self.getFormatted("float", "-1"))
        # Checks the float16 code.
        self.assertIn("= 2\n", self.getFormatted("float", "(__UINT16_TYPE__)0x4000"))
        self.assertIn("= error: unsupported byte size (1) for float format\n", self.getFormatted("float", "'a'"))

        # enumeration
        self.assertIn("= 0\n", self.getFormatted("enumeration", "0"))
        self.assertIn("= 1234567\n", self.getFormatted("enumeration", "1234567"))
        self.assertIn("= -1234567\n", self.getFormatted("enumeration", "-1234567"))

        # dec
        self.assertIn("= 1234567\n", self.getFormatted("dec", "1234567"))
        self.assertIn("= 123456789\n", self.getFormatted("dec", "(__uint128_t)123456789"))

        # unsigned decimal
        self.assertIn("= 1234567\n", self.getFormatted("unsigned decimal", "1234567"))
        self.assertIn("= 4293732729\n", self.getFormatted("unsigned decimal", "-1234567"))
        self.assertIn("= 123456789\n", self.getFormatted("unsigned decimal", "(__uint128_t)123456789"))

        # octal
        self.assertIn("= 04553207\n", self.getFormatted("octal", "1234567"))
        self.assertIn("= 0221505317046536757\n", self.getFormatted("octal", "(__uint128_t)0x123456789ABDEFull"))

        # complex float
        self.assertIn("= error: unsupported byte size (1) for complex float format\n", self.getFormatted("complex float", "'a'"))

        # complex integer
        self.assertIn("= error: unsupported byte size (1) for complex integer format\n", self.getFormatted("complex integer", "'a'"))

        # hex
        self.assertIn("= 0x00abc123\n", self.getFormatted("hex", "0xABC123"))
        self.assertIn("= 0x000000000000000000123456789abdef\n", self.getFormatted("hex", "(__uint128_t)0x123456789ABDEFull"))

        # hex float
        self.assertIn("= 0x1p1\n", self.getFormatted("hex float", "2.0f"))
        self.assertIn("= 0x1p1\n", self.getFormatted("hex float", "2.0"))
        # FIXME: long double not supported.
        self.assertIn("= error: unsupported byte size (16) for hex float format\n", self.getFormatted("hex float", "2.0l"))

        # uppercase hex
        self.assertIn("= 0x00ABC123\n", self.getFormatted("uppercase hex", "0xABC123"))

        # binary
        self.assertIn("= 0b00000000000000000000000000000010\n", self.getFormatted("binary", "2"))
        self.assertIn("= 0b01100001\n", self.getFormatted("binary", "'a'"))
        self.assertIn(" = 0b10010001101000101011001111000\n", self.getFormatted("binary", "(__uint128_t)0x12345678ll"))

        # Different character arrays.
        # FIXME: Passing a 'const char *' will ignore any given format,
        self.assertIn(r'= " \U0000001b\a\b\f\n\r\t\vaA09"', self.getFormatted("character array", "cstring"))
        self.assertIn(r'= " \U0000001b\a\b\f\n\r\t\vaA09"', self.getFormatted("c-string", "cstring"))
        self.assertIn(' = " \\e\\a\\b\\f\\n\\r\\t\\vaA09" " \\U0000001b\\a\\b\\f\\n\\r\\t\\vaA09"\n',
                      self.getFormatted("c-string", "(char *)cstring"))
        self.assertIn('=\n', self.getFormatted("c-string", "(__UINT64_TYPE__)0"))

        # Build a uint128_t that contains a series of characters in each byte.
        # First 8 byte of the uint128_t.
        cstring_chars1 = " \a\b\f\n\r\t\v"
        # Last 8 byte of the uint128_t.
        cstring_chars2 = "AZaz09\033\0"

        # Build a uint128_t value with the hex encoded characters.
        string_expr = "((__uint128_t)0x"
        for c in cstring_chars1:
             string_expr += format(ord(c), "x").zfill(2)
        string_expr += "ull << 64) | (__uint128_t)0x"
        for c in cstring_chars2:
             string_expr += format(ord(c), "x").zfill(2)
        string_expr += "ull"

        # Try to print that uint128_t with the different char formatters.
        self.assertIn('= \\0\\e90zaZA\\v\\t\\r\\n\\f\\b\\a \n', self.getFormatted("character array", string_expr))
        self.assertIn('= \\0\\e90zaZA\\v\\t\\r\\n\\f\\b\\a \n', self.getFormatted("character", string_expr))
        self.assertIn('= ..90zaZA....... \n', self.getFormatted("printable character", string_expr))
        self.assertIn('= 0x00 0x1b 0x39 0x30 0x7a 0x61 0x5a 0x41 0x0b 0x09 0x0d 0x0a 0x0c 0x08 0x07 0x20\n', self.getFormatted("unicode8", string_expr))

        # OSType
        ostype_expr = "(__UINT64_TYPE__)0x"
        for c in cstring_chars1:
             ostype_expr += format(ord(c), "x").zfill(2)
        self.assertIn("= ' \\a\\b\\f\\n\\r\\t\\v'\n", self.getFormatted("OSType", ostype_expr))

        ostype_expr = "(__UINT64_TYPE__)0x"
        for c in cstring_chars2:
             ostype_expr += format(ord(c), "x").zfill(2)
        self.assertIn("= 'AZaz09\\e\\0'\n", self.getFormatted("OSType", ostype_expr))

        self.assertIn('= 0x2007080c0a0d090b415a617a30391b00\n', self.getFormatted("OSType", string_expr))

        # bytes
        self.assertIn(r'= " \U0000001b\a\b\f\n\r\t\vaA09"', self.getFormatted("bytes", "cstring"))

        # bytes with ASCII
        self.assertIn(r'= " \U0000001b\a\b\f\n\r\t\vaA09"', self.getFormatted("bytes with ASCII", "cstring"))

        # unicode8
        self.assertIn('= 0x78 0x56 0x34 0x12\n', self.getFormatted("unicode8", "0x12345678"))

        # unicode16
        self.assertIn('= U+5678 U+1234\n', self.getFormatted("unicode16", "0x12345678"))

        # unicode32
        self.assertIn('= U+0x89abcdef U+0x01234567\n', self.getFormatted("unicode32", "(__UINT64_TYPE__)0x123456789ABCDEFll"))

        # address
        self.assertIn("= 0x00000012\n", self.getFormatted("address", "0x12"))
        self.assertIn("= 0x00000000\n", self.getFormatted("address", "0"))

        # Different fixed-width integer type arrays (e.g. 'uint8_t[]').
        self.assertIn("= {0xf8 0x56 0x34 0x12}\n", self.getFormatted("uint8_t[]", "0x123456f8"))
        self.assertIn("= {-8 86 52 18}\n", self.getFormatted("int8_t[]", "0x123456f8"))

        self.assertIn("= {0x56f8 0x1234}\n", self.getFormatted("uint16_t[]", "0x123456f8"))
        self.assertIn("= {-2312 4660}\n", self.getFormatted("int16_t[]", "0x1234F6f8"))

        self.assertIn("= {0x89abcdef 0x01234567}\n", self.getFormatted("uint32_t[]", "(__UINT64_TYPE__)0x123456789ABCDEFll"))
        self.assertIn("= {-1985229329 19088743}\n", self.getFormatted("int32_t[]", "(__UINT64_TYPE__)0x123456789ABCDEFll"))

        self.assertIn("= {0x89abcdef 0x01234567 0x00000000 0x00000000}\n", self.getFormatted("uint32_t[]", "__uint128_t i = 0x123456789ABCDEF; i"))
        self.assertIn("= {-1985229329 19088743 0 0}\n", self.getFormatted("int32_t[]", "__uint128_t i = 0x123456789ABCDEF; i"))

        self.assertIn("= {0x0123456789abcdef 0x0000000000000000}\n", self.getFormatted("uint64_t[]", "__uint128_t i = 0x123456789ABCDEF; i"))
        self.assertIn("= {-994074541749903617 0}\n", self.getFormatted("int64_t[]", "__uint128_t i = 0xF23456789ABCDEFFll; i"))

        # There is not int128_t[] style, so this only tests uint128_t[].
        self.assertIn("= {0x00000000000000000123456789abcdef}\n", self.getFormatted("uint128_t[]", "__uint128_t i = 0x123456789ABCDEF; i"))

        # Different fixed-width float type arrays.
        self.assertIn("{2 2}\n", self.getFormatted("float16[]", "0x40004000"))
        self.assertIn("{2 2}\n", self.getFormatted("float32[]", "0x4000000040000000ll"))
        self.assertIn("{2 0}\n", self.getFormatted("float64[]", "__uint128_t i = 0x4000000000000000ll; i"))

        # Invalid format string
        self.expect("expr --format invalid_format_string -- 1", error=True,
                    substrs=["error: Invalid format character or name 'invalid_format_string'. Valid values are:"])

    # Extends to host target pointer width.
    @skipIf(archs=no_match(['x86_64']))
    @no_debug_info_test
    def test_pointer(self):
        # pointer
        self.assertIn("= 0x000000000012d687\n", self.getFormatted("pointer", "1234567"))
        self.assertIn("= 0x0000000000000000\n", self.getFormatted("pointer", "0"))
        # FIXME: Just ignores the input value as it's not pointer sized.
        self.assertIn("= 0x0000000000000000\n", self.getFormatted("pointer", "'a'"))

    # Depends on the host target for decoding.
    @skipIf(archs=no_match(['x86_64']))
    @no_debug_info_test
    def test_instruction(self):
        self.assertIn("  addq   0xa(%rdi), %r8\n", self.getFormatted("instruction", "0x0a47034c"))
