"""
Test that resilient APIs work regardless of the combination of library and executable
"""
import subprocess
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess

def execute_command(command):
    # print '%% %s' % (command)
    (exit_status, output) = subprocess.getstatusoutput(command)
    return exit_status


class TestSwiftResilience(TestBase):
    # The flavors of mod store the fields of S in a different order, so the
    # rendering of a value of type S identifies the library that is really
    # loaded, independently of the one the executable was built against.
    # expect() matches substrings in the order they are given.
    G_S_SUBSTRS = {"a": ["a = 1", 's1 = "i"'], "b": ["b = 2", "a = 1"]}

    @requireNotEmbeddedSwift
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym", "dwarf"]))
    def test_cross_module_extension_a_a(self):
        """Test that LLDB can debug across resilient boundaries"""
        self.build()
        self.doTestWithFlavor("a", "a")

    @requireNotEmbeddedSwift
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym", "dwarf"]))
    def test_cross_module_extension_a_b(self):
        """Test that LLDB can debug across resilient boundaries"""
        self.build()
        self.doTestWithFlavor("a", "b")

    @requireNotEmbeddedSwift
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym", "dwarf"]))
    def test_cross_module_extension_b_a(self):
        """Test that LLDB can debug across resilient boundaries"""
        self.build()
        self.doTestWithFlavor("b", "a")

    @requireNotEmbeddedSwift
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym", "dwarf"]))
    def test_cross_module_extension_b_b(self):
        """Test that LLDB can debug across resilient boundaries"""
        self.build()
        self.doTestWithFlavor("b", "b")


    def createSymlinks(self, exe_flavor, mod_flavor):
        execute_command("cp " + self.getBuildArtifact(exe_flavor + "/main") + " " + self.getBuildArtifact("main"))
        execute_command("ln -sfn " + self.getBuildArtifact(exe_flavor + "/main.dSYM") + " " + self.getBuildArtifact("main.dSYM"))
        # The executable records the path of its serialized AST relative to
        # itself, so the module has to sit next to the copy, not next to the
        # original.
        execute_command("ln -sfn " + self.getBuildArtifact(exe_flavor + "/main.swiftmodule") + " " + self.getBuildArtifact("main.swiftmodule"))

        execute_command("cp " + self.getBuildArtifact(mod_flavor + "/libmod.dylib") + " " + self.getBuildArtifact("libmod.dylib"))
        execute_command("ln -sfn " + self.getBuildArtifact(mod_flavor + "/libmod.dylib.dSYM") + " " + self.getBuildArtifact("libmod.dylib.dSYM"))

    def cleanupSymlinks(self):
        execute_command(
            "rm -rf " +
            self.getBuildArtifact("main") + " " +
            self.getBuildArtifact("main.dSYM") + " " +
            self.getBuildArtifact("main.swiftmodule") + " " +
            self.getBuildArtifact("libmod.dylib") + " " +
            self.getBuildArtifact("libmod.dylib.dSYM"))

    def check_global(self, symbol_name, substrs):
        self.expect("target var " + symbol_name,
                    DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=substrs)
        self.expect("expr " + symbol_name,
                    DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=substrs)

    def doTestWithFlavor(self, exe_flavor, mod_flavor):
        self.createSymlinks(exe_flavor, mod_flavor)
        # Both flavors of mod are on disk under the same module name, so which
        # one SwiftASTContext imports is decided by the order in which the
        # module search paths are consulted, while reflection always describes
        # the library that is actually loaded. Cross-checking the two type
        # systems is therefore not meaningful here.
        self.runCmd("settings set symbols.swift-validate-typesystem false")

        target, process, _, breakpoint = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            exe_name=self.getBuildArtifact("main"),
            extra_images=['mod'])
        dylib_breakpoint = target.BreakpointCreateByName("fA")

        # main.swift
        self.check_global("g_main_b", ["world"])
        self.check_global("g_main_s", ["a = 1"])
        self.check_global("g_main_t", ["a = 1", "a = 1"])
        self.check_global("g_main_nested_t", ["a = 1"])
        self.check_global("g_main_c", ["a = 1"])
        self.check_global("g_main_nested_c", ["a = 1"])

        # Test defining global variables in the expression evaluator.
        self.expect("expr -- var $g_main_b = g_main_b; $g_main_b",
                    substrs=["world"])
        self.expect("expr -- var $g_main_s = S(); $g_main_s", substrs=["a = 1"])
        self.expect("expr -- var $g_main_t = (S(), S()); $g_main_t",
                    substrs=["a = 1", "a = 1"])
        self.expect("expr -- var $g_main_c = g_main_c; $g_main_c",
                    substrs=["a = 1"])

        threads = lldbutil.continue_to_breakpoint(process, dylib_breakpoint)
        self.assertTrue(len(threads) == 1)
        
        # Test global variable inside the module defining S.
        self.check_global("g_b", ["hello"])
        self.check_global("g_s", self.G_S_SUBSTRS[mod_flavor])
        self.check_global("g_t", ["a = 1", "a = 1"])
        self.check_global("g_c", ["a = 1"])
        # Test defining global variables in the expression evaluator
        # inside the module defining S.
        self.expect("expr -- var $g_b = g_b; $g_b", substrs=["hello"])
        self.expect("expr -- var $g_s = S(); $g_s", substrs=["a = 1"])
        self.expect("expr -- var $g_t = (S(), S()); $g_t",
                    substrs=["a = 1", "a = 1"])
        self.expect("expr -- var $g_c = g_c; $g_c", substrs=["a = 1"])
        threads = lldbutil.continue_to_breakpoint(process, breakpoint)

        # Back in main.swift
        self.assertTrue(len(threads) == 1)
        frame = threads[0].frames[0]
        
        # Try 'frame variable'
        var = frame.FindVariable("s")
        child = var.GetChildMemberWithName("a")
        lldbutil.check_variable(self, child, False, value="1")

        # Try the expression parser
        self.expect("expr s.a", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["1"])
        self.expect(
            "expr fA(s)",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        process.Kill()

        self.cleanupSymlinks()
