# TestSwiftInterfaceDSYM.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2019 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# -----------------------------------------------------------------------------

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftInterfaceDSYM(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    @skipIf(archs=no_match("x86_64"))
    @skipIf(debug_info=no_match(["dsym"]))
    def test_dsym_swiftinterface(self):
        """Test that LLDB can import .swiftinterface files inside a .dSYM bundle."""
        self.build()
        self.assertFalse(os.path.isdir(self.getBuildArtifact("AA.dylib.dSYM")),
                        "dylib dsym exists")

        # The custom swift module cache location
        swift_mod_cache = self.getBuildArtifact("cache")
        self.assertTrue(os.path.isfile(self.getBuildArtifact(
            "a.out.dSYM/Contents/Resources/Swift/x86_64/AA.swiftinterface")),
            "dsymutil doesn't support Swift interfaces")
        # Delete the .swiftinterface form the build dir.
        os.remove(self.getBuildArtifact("AA.swiftinterface"))

        # Clear the swift module cache (populated by the Makefile build)
        shutil.rmtree(swift_mod_cache)
        import glob
        self.assertFalse(os.path.isdir(swift_mod_cache),
                         "module cache should not exist")

        # Update the settings to use the custom module cache location
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % swift_mod_cache)

        # Set a breakpoint in and launch the main executable
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))

        # Check we are able to access the public fields of variables whose
        # types are from the .swiftinterface-only dylibs
        var = self.frame().FindVariable("x")
        lldbutil.check_variable(self, var, False, typename="AA.MyPoint")

        child_y = var.GetChildMemberWithName("y") # MyPoint.y is public
        lldbutil.check_variable(self, child_y, False, value="0")

        # MyPoint.x isn't public, but LLDB can find it through type metadata.
        child_x = var.GetChildMemberWithName("x")
        self.assertTrue(child_x.IsValid())

        # Expression evaluation using types from the .swiftinterface only
        # dylibs should work too
        lldbutil.check_expression(
            self, self.frame(), "y.magnitudeSquared", "404", use_summary=False)
        lldbutil.check_expression(
            self, self.frame(), "MyPoint(x: 1, y: 2).magnitudeSquared", "5",
            use_summary=False)

        # Check the swift module cache was populated with the .swiftmodule
        # files of the loaded modules
        self.assertTrue(os.path.isdir(swift_mod_cache), "module cache exists")
        a_modules = glob.glob(os.path.join(swift_mod_cache, 'AA-*.swiftmodule'))
        self.assertEqual(len(a_modules), 1)

    @swiftTest
    @skipIf(archs=no_match("x86_64"))
    @skipIf(debug_info=no_match(["dsym"]))
    def test_sanity_negative(self):
        """without the swiftinterface the test should fail"""
        self.build()
        self.assertFalse(os.path.isdir(self.getBuildArtifact("AA.dylib.dSYM")),
                        "dylib dsym exists")

        swift_mod_cache = self.getBuildArtifact("cache")

        # Delete the .swiftinterfaces from the .dSYM bundle.
        shutil.rmtree(
            self.getBuildArtifact("a.out.dSYM/Contents/Resources/Swift"),
            ignore_errors=True)

        # Delete the .swiftinterface form the build dir.
        os.remove(self.getBuildArtifact("AA.swiftinterface"))
        shutil.rmtree(swift_mod_cache)
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % swift_mod_cache)

        # Set a breakpoint in and launch the main executable
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))

        # We should not find a type for x.
        var = self.frame().FindVariable("x")
        # This was true for SwiftASTContext, but
        # TypeSystemSwiftTyperef succeeds, because this test it only
        # prints the type *name*.
        self.assertEqual(var.GetTypeName(), "AA.MyPoint")
        # Evaluating an expression fails, though.
        self.expect("p x", error=1)

    @swiftTest
    @skipIf(archs=no_match("x86_64"))
    @skipIf(debug_info=no_match(["dsym"]))
    def test_sanity_positive(self):
        """Test that the presence of a .swiftinterface doesn't shadow a
           .swiftmodule"""
        self.build(dictionary={'DYLIB_DSYM': 'YES'})
        self.assertTrue(os.path.isdir(self.getBuildArtifact("libAA.dylib.dSYM")),
                        "dylib dsym exists")

        swift_mod_cache = self.getBuildArtifact("cache")
        self.assertTrue(os.path.isdir("%s.dSYM/Contents/Resources/Swift"
                                      %self.getBuildArtifact()),
                        "dsymutil doesn't support Swift interfaces")

        shutil.rmtree(swift_mod_cache)
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % swift_mod_cache)

        # Set a breakpoint in and launch the main executable
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))
        self.expect('breakpoint set -f AA.swift -l 11')

        #
        # main.swift
        #

        # We *should* find a type for x.
        var = self.frame().FindVariable("x")
        lldbutil.check_variable(self, var, typename="AA.MyPoint")
        # MyPoint.x is private.
        child_x = var.GetChildMemberWithName("x")
        self.assertTrue(child_x.IsValid())

        # The expression evaluator sees the whole program, so it also
        # sees the private members.
        self.expect("p x", substrs=["x = 10"])
        # FIXME: this doesn't work, the summary/value is null/null.
        #lldbutil.check_expression(
        #    self, frame, "x", "x = 10", use_summary=False)

        #
        # AA.swift
        #

        process.Continue()
        var = self.frame().FindVariable("self")
        lldbutil.check_variable(self, var, typename="AA.MyPoint")
        # MyPoint.x is private and we should still see it.
        child_x = var.GetChildMemberWithName("x")
        lldbutil.check_variable(self, child_x, value="10")
        self.expect("p self", substrs=["x = 10"])

