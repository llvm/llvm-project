// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+fgt   < %s | FileCheck %s
// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+v8.6a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding  < %s 2>&1         | FileCheck %s --check-prefix=NOFGT

msr HFGRTR_EL2, x0
msr HFGWTR_EL2, x5
msr HFGITR_EL2, x10
msr HDFGRTR_EL2, x15
msr HDFGWTR_EL2, x20
// CHECK: msr     HFGRTR_EL2, x0          // encoding: [0x80,0x11,0x1c,0xd5]
// CHECK: msr     HFGWTR_EL2, x5          // encoding: [0xa5,0x11,0x1c,0xd5]
// CHECK: msr     HFGITR_EL2, x10         // encoding: [0xca,0x11,0x1c,0xd5]
// CHECK: msr     HDFGRTR_EL2, x15        // encoding: [0x8f,0x31,0x1c,0xd5]
// CHECK: msr     HDFGWTR_EL2, x20        // encoding: [0xb4,0x31,0x1c,0xd5]
// NOFGT: error: expected writable system register or pstate
// NOFGT: error: expected writable system register or pstate
// NOFGT: error: expected writable system register or pstate
// NOFGT: error: expected writable system register or pstate
// NOFGT: error: expected writable system register or pstate

mrs x30,  HFGRTR_EL2
mrs x25,  HFGWTR_EL2
mrs x20,  HFGITR_EL2
mrs x15,  HDFGRTR_EL2
mrs x10,  HDFGWTR_EL2
// CHECK: mrs     x30, HFGRTR_EL2         // encoding: [0x9e,0x11,0x3c,0xd5]
// CHECK: mrs     x25, HFGWTR_EL2         // encoding: [0xb9,0x11,0x3c,0xd5]
// CHECK: mrs     x20, HFGITR_EL2         // encoding: [0xd4,0x11,0x3c,0xd5]
// CHECK: mrs     x15, HDFGRTR_EL2        // encoding: [0x8f,0x31,0x3c,0xd5]
// CHECK: mrs     x10, HDFGWTR_EL2        // encoding: [0xaa,0x31,0x3c,0xd5]
// NOFGT: error: expected readable system register
// NOFGT: error: expected readable system register
// NOFGT: error: expected readable system register
// NOFGT: error: expected readable system register
// NOFGT: error: expected readable system register
