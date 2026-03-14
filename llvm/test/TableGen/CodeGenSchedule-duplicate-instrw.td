// RUN: not llvm-tblgen --gen-subtarget -I %p/../../include -I %p/Common %s -o - 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def FakeTarget : Target { }

def FakeModel : SchedMachineModel { }

def WriteA : SchedWrite;
def WriteB : SchedWrite;

let SchedModel = NoSchedModel in {
  def : InstRW<[WriteA], (instrs COPY)>;

  def : InstRW<[WriteB], (instrs COPY)>;
// CHECK: [[@LINE-1]]:3: error: Overlapping InstRW definition for "COPY" also matches previous "(instrs COPY)".
// CHECK-NEXT: def : InstRW<[WriteB], (instrs COPY)>;

// CHECK: [[@LINE-6]]:3: note: Previous match was here.
// CHECK-NEXT: def : InstRW<[WriteA], (instrs COPY)>;
}