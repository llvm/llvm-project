"""
Test lldb-dap module request
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase
import re


class TestDAP_module(lldbdap_testcase.DAPTestCaseBase):
    def run_test(self, symbol_basename, expect_debug_info_size):
        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)
        functions = ["foo"]

        # This breakpoint will be resolved only when the libfoo module is loaded
        breakpoint_ids = self.set_function_breakpoints(
            functions, wait_for_resolve=False
        )
        self.assertEqual(len(breakpoint_ids), len(functions), "expect one breakpoint")
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

        def check_symbols_loaded_with_size():
            active_modules = self.dap_server.get_modules()
            program_module = active_modules[program_basename]
            self.assertIn("symbolFilePath", program_module)
            self.assertIn(symbols_path, program_module["symbolFilePath"])
            size_regex = re.compile(r"[0-9]+(\.[0-9]*)?[KMG]?B")
            return size_regex.match(program_module["debugInfoSize"])

        if expect_debug_info_size:
            self.assertTrue(
                self.wait_until(check_symbols_loaded_with_size),
                "expect has debug info size",
            )

        active_modules = self.dap_server.get_modules()
        program_module = active_modules[program_basename]
        self.assertEqual(program_basename, program_module["name"])
        self.assertEqual(program, program_module["path"])
        self.assertIn("addressRange", program_module)

        # Collect all the module names we saw as events.
        module_new_names = []
        module_changed_names = []
        module_event = self.dap_server.wait_for_event(["module"], 1)
        while module_event is not None:
            reason = module_event["body"]["reason"]
            if reason == "new":
                module_new_names.append(module_event["body"]["module"]["name"])
            elif reason == "changed":
                module_changed_names.append(module_event["body"]["module"]["name"])

            module_event = self.dap_server.wait_for_event(["module"], 1)

        # Make sure we got an event for every active module.
        self.assertNotEqual(len(module_new_names), 0)
        for module in active_modules:
            self.assertIn(module, module_new_names)

        # Make sure we got an update event for the program module when the
        # symbols got added.
        self.assertNotEqual(len(module_changed_names), 0)
        self.assertIn(program_module["name"], module_changed_names)
        self.continue_to_exit()

    @skipIfWindows
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
    def test_modules_dsym(self):
        """
        Darwin only test with dSYM file.

        On mac, if we load a.out.dSYM as our symbol file, we will have debug symbols and we
        will have DWARF sections added to the module, so we will expect a size.
        """
        return self.run_test("a.out.dSYM", expect_debug_info_size=True)

    @skipIfWindows
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

        self.continue_to_exit()
