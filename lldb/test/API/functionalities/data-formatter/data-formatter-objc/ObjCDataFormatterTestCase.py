# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCDataFormatterTestCase(TestBase):

   mydir = TestBase.compute_mydir(__file__)

   def appkit_tester_impl(self, commands):
      self.build()
      self.appkit_common_data_formatters_command()
      commands()

   def appkit_common_data_formatters_command(self):
      """Test formatters for AppKit classes."""
      self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
          self, '// Set break point at this line.',
          lldb.SBFileSpec('main.m', False))

      # The stop reason of the thread should be breakpoint.
      self.expect(
          "thread list",
          STOPPED_DUE_TO_BREAKPOINT,
          substrs=['stopped', 'stop reason = breakpoint'])

      # This is the function to remove the custom formats in order to have a
      # clean slate for the next test case.
      def cleanup():
         self.runCmd('type format clear', check=False)
         self.runCmd('type summary clear', check=False)
         self.runCmd('type synth clear', check=False)

      # Execute the cleanup function during test case tear down.
      self.addTearDownHook(cleanup)
