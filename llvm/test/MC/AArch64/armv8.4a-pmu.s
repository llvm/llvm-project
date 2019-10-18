// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s \
// RUN: | FileCheck %s --check-prefix=CHECK

// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// ARMV8.4-A PMU
//------------------------------------------------------------------------------

// Read/Write registers:

msr PMMIR_EL1, x0
mrs x0, PMMIR_EL1

//CHECK: msr     PMMIR_EL1, x0           // encoding: [0xc0,0x9e,0x18,0xd5]
//CHECK: mrs     x0, PMMIR_EL1           // encoding: [0xc0,0x9e,0x38,0xd5]
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected readable system register
