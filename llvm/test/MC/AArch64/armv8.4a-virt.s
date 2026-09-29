// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// Virtualization Enhancements
//------------------------------------------------------------------------------

  msr   VSTCR_EL2, x0
  msr   VSTTBR_EL2, x0
  msr   SDER32_EL2, x12
  msr   CNTHVS_TVAL_EL2, x0
  msr   CNTHVS_CVAL_EL2, x0
  msr   CNTHVS_CTL_EL2, x0
  msr   CNTHPS_TVAL_EL2, x0
  msr   CNTHPS_CVAL_EL2, x0
  msr   CNTHPS_CTL_EL2, x0

//CHECK:  msr VSTCR_EL2, x0           // encoding: [0x40,0x26,0x1c,0xd5]
//CHECK:  msr VSTTBR_EL2, x0          // encoding: [0x00,0x26,0x1c,0xd5]
//CHECK:  msr SDER32_EL2, x12         // encoding: [0x2c,0x13,0x1c,0xd5]
//CHECK:  msr CNTHVS_TVAL_EL2, x0     // encoding: [0x00,0xe4,0x1c,0xd5]
//CHECK:  msr CNTHVS_CVAL_EL2, x0     // encoding: [0x40,0xe4,0x1c,0xd5]
//CHECK:  msr CNTHVS_CTL_EL2, x0      // encoding: [0x20,0xe4,0x1c,0xd5]
//CHECK:  msr CNTHPS_TVAL_EL2, x0     // encoding: [0x00,0xe5,0x1c,0xd5]
//CHECK:  msr CNTHPS_CVAL_EL2, x0     // encoding: [0x40,0xe5,0x1c,0xd5]
//CHECK:  msr CNTHPS_CTL_EL2, x0      // encoding: [0x20,0xe5,0x1c,0xd5]

//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
//CHECK-ERROR: error: expected writable system register or pstate
