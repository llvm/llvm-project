"""Show bitfields and check that they display correctly."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BitfieldsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    # BitFields exhibit crashes in record layout on Windows
    # (http://llvm.org/pr21800)
    @skipIfWindows
    def test_and_run_command(self):
        """Test 'frame variable ...' on a variable with bitfields."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        self.expect(
            "frame variable --show-types bits",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(uint32_t:1) b1 = 1',
                '(uint32_t:2) b2 = 3',
                '(uint32_t:3) b3 = 7',
                '(uint32_t) b4 = 15',
                '(uint32_t:5) b5 = 31',
                '(uint32_t:6) b6 = 63',
                '(uint32_t:7) b7 = 127',
                '(uint32_t:4) four = 15'])

        # And so should this.
        # rdar://problem/8348251
        self.expect(
            "frame variable --show-types",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(uint32_t:1) b1 = 1',
                '(uint32_t:2) b2 = 3',
                '(uint32_t:3) b3 = 7',
                '(uint32_t) b4 = 15',
                '(uint32_t:5) b5 = 31',
                '(uint32_t:6) b6 = 63',
                '(uint32_t:7) b7 = 127',
                '(uint32_t:4) four = 15'])

        self.expect("expr (bits.b1)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '1'])
        self.expect("expr (bits.b2)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '3'])
        self.expect("expr (bits.b3)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '7'])
        self.expect("expr (bits.b4)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '15'])
        self.expect("expr (bits.b5)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '31'])
        self.expect("expr (bits.b6)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '63'])
        self.expect("expr (bits.b7)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '127'])
        self.expect("expr (bits.four)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '15'])

        self.expect(
            "frame variable --show-types more_bits",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(uint32_t:3) a = 3',
                '(uint8_t:1) b = \'\\0\'',
                '(uint8_t:1) c = \'\\x01\'',
                '(uint8_t:1) d = \'\\0\''])

        self.expect("expr (more_bits.a)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', '3'])
        self.expect("expr (more_bits.b)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint8_t', '\\0'])
        self.expect("expr (more_bits.c)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint8_t', '\\x01'])
        self.expect("expr (more_bits.d)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint8_t', '\\0'])

        self.expect("expr (packed.a)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['char', "'a'"])
        self.expect("expr (packed.b)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', "10"])
        self.expect("expr/x (packed.c)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint32_t', "7112233"])

        for bit in range(1,18):
            expected = "1" if bit in [1, 5, 7, 13] else "0"
            self.expect("expr even_more_bits.b" + str(bit), VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint8_t', expected])

        for bit in [3, 10, 14]:
            self.expect("expr even_more_bits.b" + str(bit) + " = 1", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint8_t', "1"])

        self.expect(
            "frame variable --show-types even_more_bits",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(uint8_t:1) b1 = \'\\x01\'',
                '(uint8_t:1) b2 = \'\\0\'',
                '(uint8_t:1) b3 = \'\\x01\'',
                '(uint8_t:1) b4 = \'\\0\'',
                '(uint8_t:1) b5 = \'\\x01\'',
                '(uint8_t:1) b6 = \'\\0\'',
                '(uint8_t:1) b7 = \'\\x01\'',
                '(uint8_t:1) b8 = \'\\0\'',
                '(uint8_t:1) b9 = \'\\0\'',
                '(uint8_t:1) b10 = \'\\x01\'',
                '(uint8_t:1) b12 = \'\\0\'',
                '(uint8_t:1) b13 = \'\\x01\'',
                '(uint8_t:1) b14 = \'\\x01\'',
                '(uint8_t:1) b15 = \'\\0\'',
                '(uint8_t:1) b16 = \'\\0\'',
                '(uint8_t:1) b17 = \'\\0\'',
                ])

        self.expect("v/x large_packed", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["a = 0x0000000cbbbbaaaa", "b = 0x0000000dffffeee"])

    # BitFields exhibit crashes in record layout on Windows
    # (http://llvm.org/pr21800)
    @skipIfWindows
    def test_expression_bug(self):
        # Ensure evaluating (emulating) an expression does not break bitfield
        # values for already parsed variables. The expression is run twice
        # because the very first expression can resume a target (to allocate
        # memory, etc.) even if it is not being jitted.
        self.build()
        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec("main.c"),
                self.line)
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
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.c", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # The breakpoint should have a hit count of 1.
        self.assertEqual(breakpoint.GetHitCount(), 1, BREAKPOINT_HIT_ONCE)

        # Lookup the "bits" variable which contains 8 bitfields.
        frame = thread.GetFrameAtIndex(0)
        bits = frame.FindVariable("bits")
        self.DebugSBValue(bits)
        self.assertTrue(
            bits.GetTypeName() == 'Bits',
            "bits.GetTypeName() == 'Bits'")
        self.assertTrue(
            bits.GetNumChildren() == 10,
            "bits.GetNumChildren() == 10")
        test_compiler = self.getCompiler()
        self.assertTrue(bits.GetByteSize() == 32, "bits.GetByteSize() == 32")

        # Notice the pattern of int(b1.GetValue(), 0).  We pass a base of 0
        # so that the proper radix is determined based on the contents of the
        # string.
        b1 = bits.GetChildMemberWithName("b1")
        self.DebugSBValue(b1)
        self.assertTrue(b1.GetName() == "b1" and
                        b1.GetTypeName() == "uint32_t:1" and
                        b1.IsInScope() and
                        int(b1.GetValue(), 0) == 1,
                        'bits.b1 has type uint32_t:1, is in scope, and == 1')

        b7 = bits.GetChildMemberWithName("b7")
        self.DebugSBValue(b7)
        self.assertTrue(b7.GetName() == "b7" and
                        b7.GetTypeName() == "uint32_t:7" and
                        b7.IsInScope() and
                        int(b7.GetValue(), 0) == 127,
                        'bits.b7 has type uint32_t:7, is in scope, and == 127')

        four = bits.GetChildMemberWithName("four")
        self.DebugSBValue(four)
        self.assertTrue(four.GetName() == "four" and
                        four.GetTypeName() == "uint32_t:4" and
                        four.IsInScope() and
                        int(four.GetValue(), 0) == 15,
                        'bits.four has type uint32_t:4, is in scope, and == 15')

        # Now kill the process, and we are done.
        rc = target.GetProcess().Kill()
        self.assertTrue(rc.Success())
