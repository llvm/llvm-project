"""
Test case to verify stepping behavior into shared library functions.
Specifically, this tests the scenario where a function's trampoline
(PLT stub) has a different symbol at its file address,
"""

import lldb
from lldbsuite.test.decorators import skipUnlessPlatform
from lldbsuite.test.lldbtest import TestBase, line_number, configuration
import lldbsuite.test.lldbutil as lldbutil


@skipUnlessPlatform(["linux"])
class StepThroughTrampolineWithSymbol(TestBase):
    SYMBOL_NAME = "lib_add"
    FAKE_SYMBOL_NAME = "fake_lib_add"

    def test(self):
        modified_exe = self.create_modified_exe()
        (_target, _process, thread, _bkpt) = lldbutil.run_to_source_breakpoint(
            self,
            "// Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            exe_name=modified_exe,
            extra_images=["add"],
        )

        error = lldb.SBError()
        thread.StepInto(
            None, lldb.LLDB_INVALID_LINE_NUMBER, error, lldb.eOnlyDuringStepping
        )

        self.assertTrue(error.Success(), f"step into failed: {error.GetCString()}")

        # Check frame in cli.
        add_stop_line = line_number("add.c", "// End up here")
        self.expect(
            "frame info", substrs=[self.SYMBOL_NAME, "add.c:{}:".format(add_stop_line)]
        )

        # Check frame in SBAPI.
        current_frame = thread.selected_frame
        self.assertTrue(current_frame.IsValid())
        function_name = current_frame.function.name
        self.assertEqual(function_name, self.SYMBOL_NAME)

        frame_module = current_frame.module
        self.assertTrue(frame_module.IsValid())
        frame_module_name = frame_module.file.basename
        self.assertEqual(frame_module_name, "libadd.so")

    def create_modified_exe(self) -> str:
        """
        Build the executable, find the `lib_add` trampoline and add
        an the symbol `fake_lib_add` at the trampoline's file address

        Returns the modified executable.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        modulespec = lldb.SBModuleSpec()
        modulespec.SetFileSpec(lldb.SBFileSpec(exe))
        module = lldb.SBModule(modulespec)
        self.assertTrue(module.IsValid())

        add_trampoline = lldb.SBSymbol()
        for sym in module.symbols:
            if sym.name == self.SYMBOL_NAME and sym.type == lldb.eSymbolTypeTrampoline:
                add_trampoline = sym
                break

        self.assertTrue(add_trampoline.IsValid())

        # Get the trampoline's section and offset.
        add_address = add_trampoline.addr
        self.assertTrue(add_address.IsValid())

        add_section_name = add_address.section.name
        self.assertIn(".plt", add_section_name)
        add_section_offset = add_address.offset

        # Add a new symbol to the file address of lib_add trampoline.
        modified_exe = self.getBuildArtifact("mod_a.out")
        objcopy_bin = configuration.get_objcopy_path()

        build_command = [
            objcopy_bin,
            "--add-symbol",
            f"{self.FAKE_SYMBOL_NAME}={add_section_name}:{add_section_offset},function,local",
            exe,
            modified_exe,
        ]
        self.runBuildCommand(build_command)

        # Verify we added the fake symbol.
        modified_modulespec = lldb.SBModuleSpec()
        modified_modulespec.SetFileSpec(lldb.SBFileSpec(modified_exe))
        modified_module = lldb.SBModule(modified_modulespec)
        self.assertTrue(modified_module.IsValid())

        fake_symbol = modified_module.FindSymbol(self.FAKE_SYMBOL_NAME)
        self.assertTrue(fake_symbol.IsValid())

        return modified_exe
