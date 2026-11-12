// RUN: llvm-mc -triple thumb -mattr=+dotprod -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=cortex-a55 -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=cortex-a75 -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=cortex-a76 -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=cortex-a77 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=cortex-a78 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=cortex-x1 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=neoverse-n1 -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mcpu=neoverse-n2 -show-encoding < %s | FileCheck %s --check-prefix=CHECK

// RUN: not llvm-mc -triple thumb -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple thumb -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.1a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

  vudot.u8  d0, d1, d2
  vsdot.s8  d0, d1, d2
  vudot.u8  q0, q1, q4
  vsdot.s8  q0, q1, q4
  vudot.u8  d0, d1, d2[0]
  vsdot.s8  d0, d1, d2[1]
  vudot.u8  q0, q1, d4[0]
  vsdot.s8  q0, q1, d4[1]

//CHECK:  vudot.u8  d0, d1, d2      @ encoding: [0x21,0xfc,0x12,0x0d]
//CHECK:  vsdot.s8  d0, d1, d2      @ encoding: [0x21,0xfc,0x02,0x0d]
//CHECK:  vudot.u8  q0, q1, q4      @ encoding: [0x22,0xfc,0x58,0x0d]
//CHECK:  vsdot.s8  q0, q1, q4      @ encoding: [0x22,0xfc,0x48,0x0d]
//CHECK:  vudot.u8  d0, d1, d2[0]   @ encoding: [0x21,0xfe,0x12,0x0d]
//CHECK:  vsdot.s8  d0, d1, d2[1]   @ encoding: [0x21,0xfe,0x22,0x0d]
//CHECK:  vudot.u8  q0, q1, d4[0]   @ encoding: [0x22,0xfe,0x54,0x0d]
//CHECK:  vsdot.s8  q0, q1, d4[1]   @ encoding: [0x22,0xfe,0x64,0x0d]

//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod
//CHECK-ERROR: error: instruction requires: dotprod

