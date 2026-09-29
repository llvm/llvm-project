// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+ecv   < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=ERROR %s < %t
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.6a < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=ERROR %s < %t
// RUN: not llvm-mc -triple aarch64 -show-encoding               < %s 2> %t | FileCheck --check-prefix=NOECV-OUT %s
// RUN: FileCheck --check-prefix=NOECV %s < %t

// Expect no successful code generation at all with ECV disabled
// NOECV-OUT-NOT: {{msr|mrs}}

// Writable system registers
msr CNTSCALE_EL2, x1
msr CNTISCALE_EL2, x11
msr CNTPOFF_EL2, x22
msr CNTVFRQ_EL2, x3
// CHECK: msr     CNTSCALE_EL2, x1        // encoding: [0x81,0xe0,0x1c,0xd5]
// CHECK: msr     CNTISCALE_EL2, x11      // encoding: [0xab,0xe0,0x1c,0xd5]
// CHECK: msr     CNTPOFF_EL2, x22        // encoding: [0xd6,0xe0,0x1c,0xd5]
// CHECK: msr     CNTVFRQ_EL2, x3         // encoding: [0xe3,0xe0,0x1c,0xd5]
// NOECV: :[[@LINE-8]]:5: error: expected writable system register or pstate
// NOECV: :[[@LINE-8]]:5: error: expected writable system register or pstate
// NOECV: :[[@LINE-8]]:5: error: expected writable system register or pstate
// NOECV: :[[@LINE-8]]:5: error: expected writable system register or pstate

// Readonly system registers: writing them gives an error even with
// ECV enabled
msr CNTPCTSS_EL0, x13
msr CNTVCTSS_EL0, x23
// ERROR: :[[@LINE-2]]:5: error: expected writable system register or pstate
// ERROR: :[[@LINE-2]]:5: error: expected writable system register or pstate
// NOECV: :[[@LINE-4]]:5: error: expected writable system register or pstate
// NOECV: :[[@LINE-4]]:5: error: expected writable system register or pstate

mrs x0, CNTSCALE_EL2
mrs x5, CNTISCALE_EL2
mrs x10, CNTPOFF_EL2
mrs x15, CNTVFRQ_EL2
mrs x20, CNTPCTSS_EL0
mrs x30, CNTVCTSS_EL0
// CHECK: mrs     x0, CNTSCALE_EL2        // encoding: [0x80,0xe0,0x3c,0xd5]
// CHECK: mrs     x5, CNTISCALE_EL2       // encoding: [0xa5,0xe0,0x3c,0xd5]
// CHECK: mrs     x10, CNTPOFF_EL2        // encoding: [0xca,0xe0,0x3c,0xd5]
// CHECK: mrs     x15, CNTVFRQ_EL2        // encoding: [0xef,0xe0,0x3c,0xd5]
// CHECK: mrs     x20, CNTPCTSS_EL0       // encoding: [0xb4,0xe0,0x3b,0xd5]
// CHECK: mrs     x30, CNTVCTSS_EL0       // encoding: [0xde,0xe0,0x3b,0xd5]
// NOECV: :[[@LINE-12]]:9: error: expected readable system register
// NOECV: :[[@LINE-12]]:9: error: expected readable system register
// NOECV: :[[@LINE-12]]:10: error: expected readable system register
// NOECV: :[[@LINE-12]]:10: error: expected readable system register
// NOECV: :[[@LINE-12]]:10: error: expected readable system register
// NOECV: :[[@LINE-12]]:10: error: expected readable system register
