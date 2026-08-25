"""
Test the data formatters for gnustep-base's Foundation classes.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGNUstepDataFormatters(TestBase):
    def stop_at_end(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )
        # Every check below goes through the dynamic type, which is what a
        # debugger front end asks for. The setting is per target, so it has
        # to be applied to the one run_to_source_breakpoint created.
        self.runCmd("settings set target.prefer-dynamic-value run-target")

    def test_strings(self):
        """Every concrete string class prints its characters as @"..."."""
        self.stop_at_end()
        self.expect("frame variable -d run-target tinyString", substrs=['@"Hi"'])
        # A tagged tiny string goes through its own decode path rather than
        # the memory reader, so it has to escape for itself.
        self.expect("frame variable -d run-target tinyQuoted", substrs=['@"a\\"b"'])
        self.expect(
            "frame variable -d run-target constantString",
            substrs=['@"A constant string literal"'],
        )
        self.expect(
            "frame variable -d run-target unicodeConstant", substrs=['@"Grüße, 世界"']
        )
        self.expect("frame variable -d run-target emptyString", substrs=['@""'])
        self.expect("frame variable -d run-target builtString", substrs=['@"built 42"'])
        self.expect(
            "frame variable -d run-target unicodeBuilt", substrs=['@"ünïcödé 7"']
        )
        self.expect(
            "frame variable -d run-target mutableString", substrs=['@"mutable string"']
        )

    def test_numbers(self):
        """Tagged and boxed numbers print their value with a type prefix."""
        self.stop_at_end()
        self.expect("frame variable -d run-target boolYes", substrs=["YES"])
        self.expect("frame variable -d run-target smallInt", substrs=["(int)5"])
        self.expect("frame variable -d run-target taggedInt", substrs=["(long)123456"])
        self.expect("frame variable -d run-target negativeInt", substrs=["(long)-99"])
        self.expect(
            "frame variable -d run-target longLong",
            substrs=["(long)9223372036854775807"],
        )
        self.expect(
            "frame variable -d run-target unsignedLongLong",
            substrs=["(long)18446744073709551615"],
        )
        self.expect("frame variable -d run-target floatNumber", substrs=["(float)1.5"])
        self.expect(
            "frame variable -d run-target doubleNumber", substrs=["(double)3.14159"]
        )
        self.expect("frame variable -d run-target heapDouble", substrs=["(double)0.1"])

    def test_collections(self):
        """Collections summarize their count and expose elements as children."""
        self.stop_at_end()
        self.expect(
            "frame variable -d run-target emptyArray", substrs=['@"0 elements"']
        )
        self.expect("frame variable -d run-target fruits", substrs=['@"3 elements"'])
        self.expect(
            "frame variable -d run-target mutableArray", substrs=['@"4 elements"']
        )
        self.expect("frame variable -d run-target nested", substrs=['@"2 elements"'])
        self.expect(
            "frame variable -d run-target emptyDict", substrs=["0 key/value pairs"]
        )
        self.expect(
            "frame variable -d run-target person", substrs=["3 key/value pairs"]
        )
        self.expect(
            "frame variable -d run-target mutableDict", substrs=["4 key/value pairs"]
        )
        self.expect("frame variable -d run-target colors", substrs=["3 elements"])
        self.expect("frame variable -d run-target mutableSet", substrs=["4 elements"])
        self.expect("frame variable -d run-target counted", substrs=["2 elements"])

        # Children.
        self.expect(
            "frame variable -d run-target fruits[0] fruits[1] fruits[2]",
            substrs=['@"apple"', '@"banana"', '@"cherry"'],
        )
        self.expect("frame variable -d run-target nested[0]", substrs=['@"3 elements"'])
        # Dictionary entries are key/value pairs; order is hash order, so
        # look at the whole set of entries.
        self.expect(
            "frame variable -d run-target person[0] person[1] person[2]",
            # @30 lands in a tagged NSSmallInt (only -1..12 are boxed
            # singletons), which prints as a long.
            substrs=[
                "key = ",
                "value = ",
                '@"name"',
                '@"John Doe"',
                '@"age"',
                "(long)30",
                '@"skills"',
                '@"2 elements"',
            ],
            ordered=False,
        )
        self.expect(
            "frame variable -d run-target colors[0] colors[1] colors[2]",
            substrs=['@"red"', '@"green"', '@"blue"'],
            ordered=False,
        )

    def test_others(self):
        """NSData, NSDate, NSNull, nil and a custom object."""
        self.stop_at_end()
        self.expect("frame variable -d run-target data", substrs=["12 bytes"])
        self.expect(
            "frame variable -d run-target epoch", substrs=["2001-01-01 00:00:0"]
        )
        self.expect(
            "frame variable -d run-target someDate", substrs=["2023-11-14 22:13:20 UTC"]
        )
        self.expect("frame variable -d run-target null", substrs=["<null>"])
        # NSURL keeps its ivars behind GS_EXPOSE, so nothing here has debug
        # info for them; this only works by reading libobjc2's own metadata.
        self.expect(
            "frame variable -d run-target url",
            substrs=['@"https://www.gnustep.org/resources"'],
        )
        self.expect("frame variable -d run-target nilObject", substrs=["nil"])
        # A custom class: dynamic type plus formatted ivars, and its class
        # object is not itself presented as an instance.
        self.expect(
            "frame variable -d run-target anonymous",
            substrs=["(Account *) anonymous"],
        )
        self.expect(
            "frame variable -d run-target *account",
            substrs=[
                "owner = ",
                '@"Jane"',
                "balance = ",
                "(double)1234.5",
                "tags = ",
                '@"2 elements"',
            ],
        )
        self.expect(
            "frame variable -d run-target *account",
            matching=False,
            substrs=["(Account *) isa"],
        )

    def test_step_through_dispatch(self):
        """`step` at a message send lands in the method, not in objc_msgSend:
        the runtime's step-through plan resolves the implementation. Covers
        a method inside gnustep-base and one in the program."""
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// step here: Foundation", lldb.SBFileSpec("main.m")
        )
        thread.StepInto()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunctionName(), "-[GSArray count]")
        self.assertEqual(frame.GetLineEntry().GetFileSpec().GetFilename(), "GSArray.m")
        thread.StepOut()
        # Now the send to the user class.
        lldbutil.continue_to_source_breakpoint(
            self, process, "// step here: user class", lldb.SBFileSpec("main.m")
        )
        thread.StepInto()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunctionName(), "-[Account description]")
        self.assertEqual(frame.GetLineEntry().GetFileSpec().GetFilename(), "main.m")

    def test_api(self):
        """The same summaries come back through the SB API."""
        self.stop_at_end()
        frame = self.frame()
        greeting = frame.FindVariable("tinyString").GetDynamicValue(
            lldb.eDynamicCanRunTarget
        )
        self.assertEqual(greeting.GetSummary(), '@"Hi"')
        fruits = frame.FindVariable("fruits").GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertEqual(fruits.GetSummary(), '@"3 elements"')
        self.assertEqual(fruits.GetNumChildren(), 3)
        first = fruits.GetChildAtIndex(0).GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertEqual(first.GetSummary(), '@"apple"')
        person = frame.FindVariable("person").GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertEqual(person.GetNumChildren(), 3)
        pair = person.GetChildAtIndex(0)
        self.assertEqual(pair.GetChildMemberWithName("key").GetName(), "key")
        self.assertEqual(pair.GetNumChildren(), 2)

    def test_po(self):
        """`po` describes objects without gnustep-base exporting a hook.

        LLDB reproduces _NSPrintForDebugger (Source/NSDebug.m) from libobjc2's
        exported runtime API rather than resolving that symbol, which
        gnustep-base defines but does not dllexport on Windows. Nothing in
        this directory supplies it, so these assertions fail if that
        dependency ever comes back.
        """
        self.stop_at_end()
        # A user class reached through its own -description.
        self.expect("po account", substrs=["<Account Jane: 1234.5>"])
        # Tagged pointers: the receiver's class comes from the pointer bits,
        # not from a first-word read, so these only work if the description
        # call dispatches through the runtime rather than dereferencing.
        self.expect("po tinyString", substrs=["Hi"])
        self.expect("po taggedInt", substrs=["123456"])
        # A container, whose description recurses through its elements.
        self.expect("po fruits", substrs=["apple", "banana", "cherry"])
        # nil must not run anything in the inferior.
        self.expect("po nilObject", substrs=["nil"])
        # None of the above may have fallen back to `p`.
        for expr in ("account", "tinyString", "taggedInt", "fruits"):
            self.expect("po " + expr, matching=False, substrs=["was unsuccessful"])

    def test_po_is_repeatable(self):
        """Repeated `po` on the same object keeps working."""
        self.stop_at_end()
        for _ in range(3):
            self.expect("po account", substrs=["<Account Jane: 1234.5>"])
