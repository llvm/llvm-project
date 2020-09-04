"""
Test lldb-vscode setBreakpoints request
"""

from __future__ import print_function

import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import re

class TestVSCode_module(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def run_test(self, symbol_basename, expect_debug_info_size):
        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)
        functions = ['foo']
        breakpoint_ids = self.set_function_breakpoints(functions)
        self.assertEquals(len(breakpoint_ids), len(functions), 'expect one breakpoint')
        self.continue_to_breakpoints(breakpoint_ids)
        active_modules = self.vscode.get_active_modules()
        program_module = active_modules[program_basename]
        self.assertIn(program_basename, active_modules, '%s module is in active modules' % (program_basename))
        self.assertIn('name', program_module, 'make sure name is in module')
        self.assertEqual(program_basename, program_module['name'])
        self.assertIn('path', program_module, 'make sure path is in module')
        self.assertEqual(program, program_module['path'])
        self.assertTrue('symbolFilePath' not in program_module, 'Make sure a.out.stripped has no debug info')
        symbols_path = self.getBuildArtifact(symbol_basename)
        self.vscode.request_evaluate('`%s' % ('target symbols add -s "%s" "%s"' % (program, symbols_path)))

        def checkSymbolsLoadedWithSize():
            active_modules = self.vscode.get_active_modules()
            program_module = active_modules[program_basename]
            self.assertIn('symbolFilePath', program_module)
            self.assertIn(symbols_path, program_module['symbolFilePath'])
            symbol_regex = re.compile(r"[0-9]+(\.[0-9]*)?[KMG]?B")
            return symbol_regex.match(program_module['symbolStatus'])
                
        if expect_debug_info_size:
            self.waitUntil(checkSymbolsLoadedWithSize)
        active_modules = self.vscode.get_active_modules()
        program_module = active_modules[program_basename]
        self.assertEqual(program_basename, program_module['name'])
        self.assertEqual(program, program_module['path'])
        self.assertIn('addressRange', program_module)

    @skipIfWindows
    @skipUnlessDarwin
    @skipIfRemote  
    #TODO: Update the Makefile so that this test runs on Linux
    def test_module_event(self):
        '''
            Mac or linux.

            On mac, if we load a.out as our symbol file, we will use DWARF with .o files and we will
            have debug symbols, but we won't see any debug info size because all of the DWARF
            sections are in .o files.

            On other platforms, we expect a.out to have debug info, so we will expect a size.
            expect_debug_info_size = platform.system() != 'Darwin'
            return self.run_test("a.out", expect_debug_info_size)
        '''
        expect_debug_info_size = platform.system() != 'Darwin'
        return self.run_test("a.out", expect_debug_info_size)

    @skipIfWindows
    @skipUnlessDarwin
    @skipIfRemote    
    def test_module_event_dsym(self):
        '''
            Darwin only test with dSYM file.

            On mac, if we load a.out.dSYM as our symbol file, we will have debug symbols and we
            will have DWARF sections added to the module, so we will expect a size.
            return self.run_test("a.out.dSYM", True)
        '''
        return self.run_test("a.out.dSYM", True)

    @skipIfWindows
    @skipUnlessDarwin
    @skipIfRemote
    def test_compile_units(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        main_source_path = self.getSourcePath(source)
        breakpoint1_line = line_number(source, '// breakpoint 1')
        lines = [breakpoint1_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.continue_to_breakpoints(breakpoint_ids)
        moduleId = self.vscode.get_active_modules()['a.out']['id']
        response = self.vscode.request_getCompileUnits(moduleId)
        self.assertTrue(response['body'])
        self.assertTrue(len(response['body']['compileUnits']) == 1,
                        'Only one source file should exist')
        self.assertTrue(response['body']['compileUnits'][0]['compileUnitPath'] == main_source_path,
                        'Real path to main.cpp matches')

