import os

import lldb
from lldbsuite.test import configuration
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class StandaloneDebugLineTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessPlatform(["linux"])
    @skipIfLLVMTargetMissing("X86")
    def test(self):
        object_path = self.getBuildArtifact("line.o")
        self.runBuildCommand(
            [
                os.path.join(configuration.llvm_tools_dir, "llvm-mc"),
                "-triple=x86_64-pc-linux",
                "-filetype=obj",
                "-dwarf-version=2",
                self.getSourcePath("main.s"),
                "-o",
                object_path,
            ]
        )

        target = self.createTestTarget(object_path)

        # The regular symbol table advertises functions, while the standalone
        # line table must make SymbolFileDWARF the preferred reader.
        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.FindSection(".debug_line").IsValid())
        self.assertFalse(module.FindSection(".debug_info").IsValid())
        self.assertFalse(module.FindSection(".debug_abbrev").IsValid())
        self.assertTrue(module.FindSection(".symtab").IsValid())
        self.assertEqual(module.GetNumCompileUnits(), 1)

        # An address must resolve to its symbol, synthetic compile unit, and
        # line entry. It must not manufacture a DWARF function.
        symbol = module.FindSymbol("foo", lldb.eSymbolTypeCode)
        self.assertTrue(symbol.IsValid())

        address = symbol.GetStartAddress()
        self.assertTrue(address.IsValid())
        self.assertEqual(address.GetSymbol().GetName(), "foo")
        self.assertFalse(address.GetFunction().IsValid())

        compile_unit = address.GetCompileUnit()
        self.assertTrue(compile_unit.IsValid())
        self.assertEqual(compile_unit.GetFileSpec().GetFilename(), "standalone.c")

        line_entry = address.GetLineEntry()
        self.assertTrue(line_entry.IsValid())
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "standalone.c")
        self.assertEqual(line_entry.GetLine(), 42)
        self.assertEqual(line_entry.GetColumn(), 7)

        # Source-to-address lookup must use the same standalone line table.
        breakpoint = target.BreakpointCreateByLocation("standalone.c", 42)
        self.assertEqual(breakpoint.GetNumLocations(), 1)
        breakpoint_address = breakpoint.GetLocationAtIndex(0).GetAddress()
        self.assertEqual(breakpoint_address, address)
