// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+bti  < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding              < %s | FileCheck %s --check-prefix=HINT

// "bti r" is now the preferred disassembly, instead of plain `bti`
// Check the disassembly when either is specified

bti
bti r

// CHECK: bti r     // encoding: [0x1f,0x24,0x03,0xd5]
// CHECK: bti r     // encoding: [0x1f,0x24,0x03,0xd5]

// HINT: hint #32   // encoding: [0x1f,0x24,0x03,0xd5]
// HINT: hint #32   // encoding: [0x1f,0x24,0x03,0xd5]

hint #32

// CHECK: bti r     // encoding: [0x1f,0x24,0x03,0xd5]
// HINT: hint #32   // encoding: [0x1f,0x24,0x03,0xd5]
