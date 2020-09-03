// RUN: llvm-mc -triple arm -mattr=+dotprod -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=cortex-a55 -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=cortex-a75 -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=cortex-a76 -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=neoverse-n1 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=cortex-a77 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=cortex-a78 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mcpu=cortex-x1 -show-encoding < %s | FileCheck %s --check-prefix=CHECK

// RUN: not llvm-mc -triple arm -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple arm -mcpu=cortex-a77 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple arm -mcpu=cortex-a78 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple arm -mcpu=cortex-x1 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple arm -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple arm -mattr=+v8.1a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple arm -mattr=+v8.2a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s

vudot.u8 d0, d1, d2
vsdot.s8 d0, d1, d2
vudot.u8 q0, q1, q4
vsdot.s8 q0, q1, q4
vudot.u8 d0, d1, d2[0]
vsdot.s8 d0, d1, d2[1]
vudot.u8 q0, q1, d4[0]
vsdot.s8 q0, q1, d4[1]

// CHECK: vudot.u8  d0, d1, d2      @ encoding: [0x12,0x0d,0x21,0xfc]
// CHECK: vsdot.s8  d0, d1, d2      @ encoding: [0x02,0x0d,0x21,0xfc]
// CHECK: vudot.u8  q0, q1, q4      @ encoding: [0x58,0x0d,0x22,0xfc]
// CHECK: vsdot.s8  q0, q1, q4      @ encoding: [0x48,0x0d,0x22,0xfc]
// CHECK: vudot.u8  d0, d1, d2[0]   @ encoding: [0x12,0x0d,0x21,0xfe]
// CHECK: vsdot.s8  d0, d1, d2[1]   @ encoding: [0x22,0x0d,0x21,0xfe]
// CHECK: vudot.u8  q0, q1, d4[0]   @ encoding: [0x54,0x0d,0x22,0xfe]
// CHECK: vsdot.s8  q0, q1, d4[1]   @ encoding: [0x64,0x0d,0x22,0xfe]

// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
