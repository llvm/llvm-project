// ETE System registers
//
// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

// Read from system register
mrs x0, TRCRSR
mrs x0, TRCEXTINSELR
mrs x0, TRCEXTINSELR0
mrs x0, TRCEXTINSELR1
mrs x0, TRCEXTINSELR2
mrs x0, TRCEXTINSELR3

// CHECK: mrs x0, TRCRSR        // encoding: [0x00,0x0a,0x31,0xd5]
// CHECK: mrs x0, TRCEXTINSELR  // encoding: [0x80,0x08,0x31,0xd5]
// CHECK: mrs x0, TRCEXTINSELR  // encoding: [0x80,0x08,0x31,0xd5]
// CHECK: mrs x0, TRCEXTINSELR1 // encoding: [0x80,0x09,0x31,0xd5]
// CHECK: mrs x0, TRCEXTINSELR2 // encoding: [0x80,0x0a,0x31,0xd5]
// CHECK: mrs x0, TRCEXTINSELR3 // encoding: [0x80,0x0b,0x31,0xd5]

// Write to system register
msr TRCRSR, x0
msr TRCEXTINSELR,  x0
msr TRCEXTINSELR0, x0
msr TRCEXTINSELR1, x0
msr TRCEXTINSELR2, x0
msr TRCEXTINSELR3, x0

// CHECK: msr TRCRSR, x0        // encoding: [0x00,0x0a,0x11,0xd5]
// CHECK: msr TRCEXTINSELR, x0  // encoding: [0x80,0x08,0x11,0xd5]
// CHECK: msr TRCEXTINSELR, x0  // encoding: [0x80,0x08,0x11,0xd5]
// CHECK: msr TRCEXTINSELR1, x0 // encoding: [0x80,0x09,0x11,0xd5]
// CHECK: msr TRCEXTINSELR2, x0 // encoding: [0x80,0x0a,0x11,0xd5]
// CHECK: msr TRCEXTINSELR3, x0 // encoding: [0x80,0x0b,0x11,0xd5]
