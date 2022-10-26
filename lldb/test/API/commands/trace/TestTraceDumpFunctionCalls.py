from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestTraceDumpInfo(TraceIntelPTTestCaseBase):
    def testDumpSimpleFunctionCalls(self):
      self.expect("trace load -v " +
        os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"))

      self.expect("thread trace dump function-calls 2",
        error=True, substrs=['error: no thread with index: "2"'])

      self.expect("thread trace dump function-calls 1 -j",
        substrs=['[{"tracedSegments":[{"firstInstructionId":"3","lastInstructionId":"26"}]}]'])

      self.expect("thread trace dump function-calls 1 -J",
        substrs=['''[
  {
    "tracedSegments": [
      {
        "firstInstructionId": "3",
        "lastInstructionId": "26"
      }
    ]
  }
]'''])

      # We test first some code without function call
      self.expect("thread trace dump function-calls 1",
        substrs=['''thread #1: tid = 3842849

[call tree #0]
a.out`main + 4 at main.cpp:2 to 4:0  [3, 26]'''])

    def testFunctionCallsWithErrors(self):
      self.expect("trace load -v " +
        os.path.join(self.getSourceDir(), "intelpt-multi-core-trace", "trace.json"))

      # We expect that tracing errors appear as a different tree
      self.expect("thread trace dump function-calls 2",
        substrs=['''thread #2: tid = 3497496

[call tree #0]
m.out`foo() + 65 at multi_thread.cpp:12:21 to 12:21  [4, 19524]

[call tree #1]
<tracing errors>  [19532, 19532]'''])

      self.expect("thread trace dump function-calls 2 -J",
        substrs=['''[
  {
    "tracedSegments": [
      {
        "firstInstructionId": "4",
        "lastInstructionId": "19524"
      }
    ]
  },
  {
    "tracedSegments": [
      {
        "firstInstructionId": "19532",
        "lastInstructionId": "19532"
      }
    ]
  }
]'''])

      self.expect("thread trace dump function-calls 3",
        substrs=['''thread #3: tid = 3497497

[call tree #0]
m.out`bar() + 30 at multi_thread.cpp:19:3 to 20:6  [5, 61831]

[call tree #1]
<tracing errors>  [61834, 61834]'''])

      self.expect("thread trace dump function-calls 3 -J",
        substrs=['''[
  {
    "tracedSegments": [
      {
        "firstInstructionId": "5",
        "lastInstructionId": "61831"
      }
    ]
  },
  {
    "tracedSegments": [
      {
        "firstInstructionId": "61834",
        "lastInstructionId": "61834"
      }
    ]
  }
]'''])

    def testInlineFunctionCalls(self):
      self.expect("file " + os.path.join(self.getSourceDir(), "inline-function", "a.out"))
      self.expect("b main") # we'll trace from the beginning of main
      self.expect("b 17")
      self.expect("r")
      self.expect("thread trace start")
      self.expect("c")
      self.expect("thread trace dump function-calls",
        substrs=['''[call tree #0]
a.out`main + 8 at inline.cpp:15:7 to 16:14  [2, 6]
  a.out`foo(int) at inline.cpp:8:16 to 9:15  [7, 14]
    a.out`foo(int) + 22 [inlined] mult(int, int) at inline.cpp:2:7 to 5:10  [15, 22]
  a.out`foo(int) + 49 at inline.cpp:9:15 to 12:1  [23, 27]
a.out`main + 25 at inline.cpp:16:14 to 16:14  [28, 28]'''])

      self.expect("thread trace dump function-calls -J",
        substrs=['''[
  {
    "tracedSegments": [
      {
        "firstInstructionId": "2",
        "lastInstructionId": "6",
        "nestedCall": {
          "tracedSegments": [
            {
              "firstInstructionId": "7",
              "lastInstructionId": "14",
              "nestedCall": {
                "tracedSegments": [
                  {
                    "firstInstructionId": "15",
                    "lastInstructionId": "22"
                  }
                ]
              }
            },
            {
              "firstInstructionId": "23",
              "lastInstructionId": "27"
            }
          ]
        }
      },
      {
        "firstInstructionId": "28",
        "lastInstructionId": "28"
      }
    ]
  }
]'''])

    def testIncompleteInlineFunctionCalls(self):
      self.expect("file " + os.path.join(self.getSourceDir(), "inline-function", "a.out"))
      self.expect("b 4") # we'll trace from the middle of the inline method
      self.expect("b 17")
      self.expect("r")
      self.expect("thread trace start")
      self.expect("c")
      self.expect("thread trace dump function-calls",
        substrs=['''[call tree #0]
a.out`main
  a.out`foo(int)
    a.out`foo(int) + 36 [inlined] mult(int, int) + 14 at inline.cpp:4:5 to 5:10  [2, 6]
  a.out`foo(int) + 49 at inline.cpp:9:15 to 12:1  [7, 11]
a.out`main + 25 at inline.cpp:16:14 to 16:14  [12, 12]'''])

      self.expect("thread trace dump function-calls -J",
        substrs=['''[
  {
    "untracedPrefixSegment": {
      "nestedCall": {
        "untracedPrefixSegment": {
          "nestedCall": {
            "tracedSegments": [
              {
                "firstInstructionId": "2",
                "lastInstructionId": "6"
              }
            ]
          }
        },
        "tracedSegments": [
          {
            "firstInstructionId": "7",
            "lastInstructionId": "11"
          }
        ]
      }
    },
    "tracedSegments": [
      {
        "firstInstructionId": "12",
        "lastInstructionId": "12"
      }
    ]
  }
]'''])

    def testMultifileFunctionCalls(self):
      # This test is extremely important because it first calls the method foo() which requires going through the dynamic linking.
      # You'll see the entry "a.out`symbol stub for: foo()" which will invoke the ld linker, which will in turn find the actual foo
      # function and eventually invoke it.  However, we don't have the image of the linker in the trace bundle, so we'll see errors
      # because the decoder couldn't find the linker binary! After those failures, the linker will resume right where we return to
      # main after foo() finished.
      # Then, we call foo() again, but because it has already been loaded by the linker, we don't invoke the linker anymore! And
      # we'll see a nice tree without errors in this second invocation. Something interesting happens here. We still have an
      # invocation to the symbol stub for foo(), but it modifies the stack so that when we return from foo() we don't stop again
      # at the symbol stub, but instead we return directly to main. This is an example of returning several levels up in the
      # call stack.
      # Not only that, we also have an inline method in between.
      self.expect("trace load " + os.path.join(self.getSourceDir(), "intelpt-trace-multi-file", "multi-file-no-ld.json"))
      self.expect("thread trace dump function-calls",
        substrs=['''thread #1: tid = 815455

[call tree #0]
a.out`main + 15 at main.cpp:10 to 10:0  [3, 3]
  a.out`symbol stub for: foo() to <+11>  [7, 9]
    a.out`a.out[0x0000000000400510] to a.out[0x0000000000400516]  [10, 11]

[call tree #1]
<tracing errors>  [12, 12]

[call tree #2]
a.out`main + 20 at main.cpp:10 to 12:0  [16, 22]
  a.out`main + 34 [inlined] inline_function() at main.cpp:4 to 6:0  [26, 30]
a.out`main + 55 at main.cpp:14 to 16:0  [31, 37]
  a.out`symbol stub for: foo() to <+0>  [38, 38]
    libfoo.so`foo() at foo.cpp:3 to 4:0  [39, 42]
      libfoo.so`symbol stub for: bar() to <+0>  [43, 43]
        libbar.so`bar() at bar.cpp:1 to 4:0  [44, 52]
    libfoo.so`foo() + 13 at foo.cpp:4 to 6:0  [53, 60]
a.out`main + 68 at main.cpp:16 to 16:0  [61, 63]'''])

      self.expect("thread trace dump function-calls -J",
        substrs=['''[
  {
    "tracedSegments": [
      {
        "firstInstructionId": "3",
        "lastInstructionId": "3",
        "nestedCall": {
          "tracedSegments": [
            {
              "firstInstructionId": "7",
              "lastInstructionId": "9",
              "nestedCall": {
                "tracedSegments": [
                  {
                    "firstInstructionId": "10",
                    "lastInstructionId": "11"
                  }
                ]
              }
            }
          ]
        }
      }
    ]
  },
  {
    "tracedSegments": [
      {
        "firstInstructionId": "12",
        "lastInstructionId": "12"
      }
    ]
  },
  {
    "tracedSegments": [
      {
        "firstInstructionId": "16",
        "lastInstructionId": "22",
        "nestedCall": {
          "tracedSegments": [
            {
              "firstInstructionId": "26",
              "lastInstructionId": "30"
            }
          ]
        }
      },
      {
        "firstInstructionId": "31",
        "lastInstructionId": "37",
        "nestedCall": {
          "tracedSegments": [
            {
              "firstInstructionId": "38",
              "lastInstructionId": "38",
              "nestedCall": {
                "tracedSegments": [
                  {
                    "firstInstructionId": "39",
                    "lastInstructionId": "42",
                    "nestedCall": {
                      "tracedSegments": [
                        {
                          "firstInstructionId": "43",
                          "lastInstructionId": "43",
                          "nestedCall": {
                            "tracedSegments": [
                              {
                                "firstInstructionId": "44",
                                "lastInstructionId": "52"
                              }
                            ]
                          }
                        }
                      ]
                    }
                  },
                  {
                    "firstInstructionId": "53",
                    "lastInstructionId": "60"
                  }
                ]
              }
            }
          ]
        }
      },
      {
        "firstInstructionId": "61",
        "lastInstructionId": "63"
      }
    ]
  }
]'''])
