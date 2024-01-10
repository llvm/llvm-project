"""
Test lldb-dap setBreakpoints request
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import re


class TestDAP_module(lldbdap_testcase.DAPTestCaseBase):
    def run_test(self, symbol_basename, expect_debug_info_size):
        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)
        functions = ["foo"]
        breakpoint_ids = self.set_function_breakpoints(functions)
        self.assertEquals(len(breakpoint_ids), len(functions), "expect one breakpoint")
        self.continue_to_breakpoints(breakpoint_ids)
        active_modules = self.dap_server.get_modules()
        program_module = active_modules[program_basename]
        self.assertIn(
            program_basename,
            active_modules,
            "%s module is in active modules" % (program_basename),
        )
        self.assertIn("name", program_module, "make sure name is in module")
        self.assertEqual(program_basename, program_module["name"])
        self.assertIn("path", program_module, "make sure path is in module")
        self.assertEqual(program, program_module["path"])
        self.assertNotIn(
            "symbolFilePath",
            program_module,
            "Make sure a.out.stripped has no debug info",
        )
        symbols_path = self.getBuildArtifact(symbol_basename)
        self.dap_server.request_evaluate(
            "`%s" % ('target symbols add -s "%s" "%s"' % (program, symbols_path)),
            context="repl",
        )

        def checkSymbolsLoadedWithSize():
            active_modules = self.dap_server.get_modules()
            program_module = active_modules[program_basename]
            self.assertIn("symbolFilePath", program_module)
            self.assertIn(symbols_path, program_module["symbolFilePath"])
            symbol_regex = re.compile(r"[0-9]+(\.[0-9]*)?[KMG]?B")
            return symbol_regex.match(program_module["symbolStatus"])

        if expect_debug_info_size:
            self.waitUntil(checkSymbolsLoadedWithSize)
        active_modules = self.dap_server.get_modules()
        program_module = active_modules[program_basename]
        self.assertEqual(program_basename, program_module["name"])
        self.assertEqual(program, program_module["path"])
        self.assertIn("addressRange", program_module)

    @skipIfWindows
    @skipIfRemote
    def test_modules(self):
        """
        Mac or linux.

        On mac, if we load a.out as our symbol file, we will use DWARF with .o files and we will
        have debug symbols, but we won't see any debug info size because all of the DWARF
        sections are in .o files.

        On other platforms, we expect a.out to have debug info, so we will expect a size.
        """
        return self.run_test(
            "a.out", expect_debug_info_size=platform.system() != "Darwin"
        )

    @skipUnlessDarwin
    @skipIfRemote
    def test_modules_dsym(self):
        """
        Darwin only test with dSYM file.

        On mac, if we load a.out.dSYM as our symbol file, we will have debug symbols and we
        will have DWARF sections added to the module, so we will expect a size.
        """
        return self.run_test("a.out.dSYM", expect_debug_info_size=True)

    @skipIfWindows
    @skipIfRemote
    def test_compile_units(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        main_source_path = self.getSourcePath(source)
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.continue_to_breakpoints(breakpoint_ids)
        moduleId = self.dap_server.get_modules()["a.out"]["id"]
        response = self.dap_server.request_compileUnits(moduleId)
        self.assertTrue(response["body"])
        cu_paths = [cu["compileUnitPath"] for cu in response["body"]["compileUnits"]]
        self.assertIn(main_source_path, cu_paths, "Real path to main.cpp matches")
