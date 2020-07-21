// RUN: llvm-mc -triple aarch64 -mattr=+dotprod -show-encoding < %s | FileCheck %s  --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a55 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a65 -show-encoding < %s| FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a65ae -show-encoding < %s| FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a75 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a76 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a77 -show-encoding < %s     | FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-a78 -show-encoding < %s     | FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=cortex-x1 -show-encoding < %s     | FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=neoverse-e1 -show-encoding < %s| FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=neoverse-n1 -show-encoding < %s| FileCheck %s --check-prefix=CHECK-DOTPROD
// RUN: llvm-mc -triple aarch64 -mcpu=tsv110 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-DOTPROD

// RUN: not llvm-mc -triple aarch64 -mattr=+v8.2a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=cortex-a65 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=cortex-a65ae -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=neoverse-e1 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=cortex-a77 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=cortex-a78 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=cortex-x1 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s
// RUN: not llvm-mc -triple aarch64 -mcpu=neoverse-n1 -mattr=-dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DOTPROD < %t %s

udot v0.2s, v1.8b, v2.8b
sdot v0.2s, v1.8b, v2.8b
udot v0.4s, v1.16b, v2.16b
sdot v0.4s, v1.16b, v2.16b
udot v0.2s, v1.8b, v2.4b[0]
sdot v0.2s, v1.8b, v2.4b[1]
udot v0.4s, v1.16b, v2.4b[2]
sdot v0.4s, v1.16b, v2.4b[3]

// Check that the upper case types are aliases
udot v0.2S, v1.8B, v2.4B[0]
udot v0.4S, v1.16B, v2.4B[2]

// CHECK-DOTPROD:  udot  v0.2s, v1.8b, v2.8b     // encoding: [0x20,0x94,0x82,0x2e]
// CHECK-DOTPROD:  sdot  v0.2s, v1.8b, v2.8b     // encoding: [0x20,0x94,0x82,0x0e]
// CHECK-DOTPROD:  udot  v0.4s, v1.16b, v2.16b   // encoding: [0x20,0x94,0x82,0x6e]
// CHECK-DOTPROD:  sdot  v0.4s, v1.16b, v2.16b   // encoding: [0x20,0x94,0x82,0x4e]
// CHECK-DOTPROD:  udot  v0.2s, v1.8b, v2.4b[0]  // encoding: [0x20,0xe0,0x82,0x2f]
// CHECK-DOTPROD:  sdot  v0.2s, v1.8b, v2.4b[1]  // encoding: [0x20,0xe0,0xa2,0x0f]
// CHECK-DOTPROD:  udot  v0.4s, v1.16b, v2.4b[2] // encoding: [0x20,0xe8,0x82,0x6f]
// CHECK-DOTPROD:  sdot  v0.4s, v1.16b, v2.4b[3] // encoding: [0x20,0xe8,0xa2,0x4f]

// CHECK-DOTPROD:  udot  v0.2s, v1.8b, v2.4b[0]  // encoding: [0x20,0xe0,0x82,0x2f]
// CHECK-DOTPROD:  udot  v0.4s, v1.16b, v2.4b[2] // encoding: [0x20,0xe8,0x82,0x6f]

// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: udot v0.2s, v1.8b, v2.8b
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: sdot v0.2s, v1.8b, v2.8b
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: udot v0.4s, v1.16b, v2.16b
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: sdot v0.4s, v1.16b, v2.16b
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: udot v0.2s, v1.8b, v2.4b[0]
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: sdot v0.2s, v1.8b, v2.4b[1]
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: udot v0.4s, v1.16b, v2.4b[2]
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: sdot v0.4s, v1.16b, v2.4b[3]
// CHECK-NO-DOTPROD: ^

// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: udot v0.2S, v1.8B, v2.4B[0]
// CHECK-NO-DOTPROD: ^
// CHECK-NO-DOTPROD: error: instruction requires: dotprod
// CHECK-NO-DOTPROD: udot v0.4S, v1.16B, v2.4B[2]
// CHECK-NO-DOTPROD: ^
