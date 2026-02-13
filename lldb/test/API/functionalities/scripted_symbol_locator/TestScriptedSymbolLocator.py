"""
Test the ScriptedSymbolLocator plugin for source file resolution.
"""

import os
import shutil
import tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ScriptedSymbolLocatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")

    def import_locator(self):
        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "source_locator.py")
        )

    def set_locator_class(self, class_name):
        self.runCmd(
            "settings set plugin.symbol-locator.scripted.script-class " + class_name
        )

    def clear_locator_class(self):
        self.runCmd('settings set plugin.symbol-locator.scripted.script-class ""')

    def script(self, expr):
        """Execute a Python expression in LLDB's script interpreter and return
        the result as a string."""
        ret = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "script " + expr, ret
        )
        return ret.GetOutput().strip() if ret.Succeeded() else ""

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_locate_source_file(self):
        """Test that the scripted locator resolves source files and receives
        an SBModule with a valid UUID."""
        self.build()

        # Copy main.c to a temp directory so the locator can "resolve" to it.
        tmp_dir = tempfile.mkdtemp()
        self.addTearDownHook(lambda: shutil.rmtree(tmp_dir))
        shutil.copy(os.path.join(self.getSourceDir(), "main.c"), tmp_dir)

        # Create the target BEFORE setting the script class, so module loading
        # (which may run on worker threads) does not trigger the Python locator.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target and target.IsValid(), VALID_TARGET)

        # Now set up the scripted locator. LocateSourceFile will only be called
        # from the main thread when we access a frame's line entry.
        self.import_locator()
        self.script(
            "source_locator.SourceLocator.resolved_dir = '%s'" % tmp_dir
        )
        self.set_locator_class("source_locator.SourceLocator")
        self.addTearDownHook(lambda: self.clear_locator_class())

        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1)

        # Launch and stop at the breakpoint so ApplyFileMappings runs on
        # the main thread via StackFrame::GetSymbolContext.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertIsNotNone(process)
        self.assertState(process.GetState(), lldb.eStateStopped)

        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        line_entry = frame.GetLineEntry()
        self.assertTrue(line_entry and line_entry.IsValid(), "Line entry is valid")
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "main.c")

        # Verify the resolved path points to our temp directory.
        resolved_dir = line_entry.GetFileSpec().GetDirectory()
        self.assertEqual(resolved_dir, tmp_dir)

        # Verify the locator was called with a valid UUID by reading calls
        # from LLDB's Python namespace.
        calls_str = self.script("source_locator.SourceLocator.calls")
        self.assertIn("main.c", calls_str, "Locator should have been called")

        self.dbg.DeleteTarget(target)

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_locate_source_file_none_fallthrough(self):
        """Test that returning None falls through to normal LLDB resolution,
        and that having no script class set also works normally."""
        self.build()

        # First: test with NoneLocator -- should fall through.
        self.import_locator()
        self.set_locator_class("source_locator.NoneLocator")
        self.addTearDownHook(lambda: self.clear_locator_class())

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target and target.IsValid(), VALID_TARGET)

        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1)

        loc = bp.GetLocationAtIndex(0)
        line_entry = loc.GetAddress().GetLineEntry()
        self.assertTrue(line_entry and line_entry.IsValid(), "Line entry is valid")
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "main.c")

        self.dbg.DeleteTarget(target)

        # Second: test with no script class set -- should also work normally.
        self.clear_locator_class()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target and target.IsValid(), VALID_TARGET)

        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1)

        loc = bp.GetLocationAtIndex(0)
        line_entry = loc.GetAddress().GetLineEntry()
        self.assertTrue(line_entry and line_entry.IsValid(), "Line entry is valid")
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "main.c")

        self.dbg.DeleteTarget(target)

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_invalid_script_class(self):
        """Test that an invalid script class name is handled gracefully
        without crashing, and breakpoints still resolve."""
        self.build()

        self.set_locator_class("nonexistent_module.NonexistentClass")
        self.addTearDownHook(lambda: self.clear_locator_class())

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target and target.IsValid(), VALID_TARGET)

        # Should not crash -- breakpoint should still resolve via normal path.
        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1)

        loc = bp.GetLocationAtIndex(0)
        line_entry = loc.GetAddress().GetLineEntry()
        self.assertTrue(line_entry and line_entry.IsValid(), "Line entry is valid")

        self.dbg.DeleteTarget(target)
