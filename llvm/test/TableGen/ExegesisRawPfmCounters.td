// RUN: llvm-tblgen --gen-exegesis -I %p/../../include %s | FileCheck %s
// CHECK-LABEL: static const PfmCountersInfo TestTargetTestPfmCounters = {
// CHECK-NEXT:  nullptr, // Cycle counter uses raw encoding below.
// CHECK-NEXT:  0x003c, 0x00, // Raw cycle counter
// CHECK-NEXT:  nullptr, // Uops counter uses raw encoding below.
// CHECK-NEXT:  0x00c2, 0x01, // Raw uops counter
// CHECK-NEXT:  nullptr, 0, // No issue counters
// CHECK-NEXT:  nullptr, 0 // No validation counters.
// CHECK-NEXT: };

include "llvm/Target/TargetPfmCounters.td"

// Fake target for ExegesisEmitter
class Target {}
def TestTarget : Target {}

def TestRawCycleCounter : PfmCounter<"testCycles"> {
  let EventSelect = 0x003c;
  let UMask = 0x00;
}

def TestRawUopsCounter : PfmCounter<"testUops"> {
  let EventSelect = 0x00c2;
  let UMask = 0x01;
}

def TestPfmCounters : ProcPfmCounters {
  let CycleCounter = TestRawCycleCounter;
  let UopsCounter = TestRawUopsCounter;
}

def : PfmCountersDefaultBinding<TestPfmCounters>;
