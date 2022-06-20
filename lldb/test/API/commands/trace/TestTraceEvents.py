import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceEvents(TraceIntelPTTestCaseBase):

    @testSBAPIAndCommands
    def testPauseEvents(self):
      '''
        Everytime the target stops running on the CPU, a 'disabled' event will
        be emitted, which is represented by the TraceCursor API as a 'paused'
        event.
      '''
      self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace-multi-file", "a.out"))
      self.expect("b 12")
      self.expect("r")
      self.traceStartThread()
      self.expect("n")
      self.expect("n")
      self.expect("si")
      self.expect("si")
      self.expect("si")
      # We ensure that the paused events are printed correctly forward
      self.expect("thread trace dump instructions -e -f",
        patterns=[f'''thread #1: tid = .*
  a.out`main \+ 23 at main.cpp:12
    0: {ADDRESS_REGEX}    movl .*
  \[paused\]
    1: {ADDRESS_REGEX}    addl .*
    2: {ADDRESS_REGEX}    movl .*
  \[paused\]
  a.out`main \+ 34 \[inlined\] inline_function\(\) at main.cpp:4
    3: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 41 \[inlined\] inline_function\(\) \+ 7 at main.cpp:5
    4: {ADDRESS_REGEX}    movl .*
    5: {ADDRESS_REGEX}    addl .*
    6: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 52 \[inlined\] inline_function\(\) \+ 18 at main.cpp:6
    7: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 55 at main.cpp:14
    8: {ADDRESS_REGEX}    movl .*
    9: {ADDRESS_REGEX}    addl .*
    10: {ADDRESS_REGEX}    movl .*
  \[paused\]
  a.out`main \+ 63 at main.cpp:16
    11: {ADDRESS_REGEX}    callq  .* ; symbol stub for: foo\(\)
  \[paused\]
  a.out`symbol stub for: foo\(\)
    12: {ADDRESS_REGEX}    jmpq'''])

      # We ensure that the paused events are printed correctly backward
      self.expect("thread trace dump instructions -e --id 12",
        patterns=[f'''thread #1: tid = .*
  a.out`symbol stub for: foo\(\)
    12: {ADDRESS_REGEX}    jmpq .*
  \[paused\]
  a.out`main \+ 63 at main.cpp:16
    11: {ADDRESS_REGEX}    callq  .* ; symbol stub for: foo\(\)
  \[paused\]
  a.out`main \+ 60 at main.cpp:14
    10: {ADDRESS_REGEX}    movl .*
    9: {ADDRESS_REGEX}    addl .*
    8: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 52 \[inlined\] inline_function\(\) \+ 18 at main.cpp:6
    7: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 49 \[inlined\] inline_function\(\) \+ 15 at main.cpp:5
    6: {ADDRESS_REGEX}    movl .*
    5: {ADDRESS_REGEX}    addl .*
    4: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 34 \[inlined\] inline_function\(\) at main.cpp:4
    3: {ADDRESS_REGEX}    movl .*
  \[paused\]
  a.out`main \+ 31 at main.cpp:12
    2: {ADDRESS_REGEX}    movl .*
    1: {ADDRESS_REGEX}    addl .*
  \[paused\]
    0: {ADDRESS_REGEX}    movl .*'''])
