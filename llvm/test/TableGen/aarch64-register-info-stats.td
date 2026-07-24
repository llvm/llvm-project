// REQUIRES: asserts
// RUN: llvm-tblgen -gen-register-info -stats \
// RUN:   -I %p/../../lib/Target/AArch64 -I %p/../../include \
// RUN:   %s -o %t.inc 2>&1 | FileCheck %s

// Keep changes to the size of AArch64's generated register model visible.

include "llvm/Target/Target.td"
include "AArch64RegisterInfo.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
}

// CHECK-DAG: 83 register-info-emitter - Number of explicit register classes
// CHECK-DAG: 447 register-info-emitter - Number of synthesized register classes
// CHECK-DAG: 190 register-info-emitter - Number of register pressure sets
