// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+btie < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+bti  < %s | FileCheck %s --check-prefix=NOBTIE
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding              < %s | FileCheck %s --check-prefix=HINT

// "bti r" is the preferred encoding when +btie or +poe2 is specified.
// Otherwise default back to plain "bti". They are aliases of each other.
// Check that disassembly when `btie` is not specified causes plain
// "bti" to be emitted.

bti
bti r

// CHECK: bti r     // encoding: [0x1f,0x24,0x03,0xd5]
// CHECK: bti r     // encoding: [0x1f,0x24,0x03,0xd5]

// NOBTIE: bti      // encoding: [0x1f,0x24,0x03,0xd5]
// NOBTIE: bti      // encoding: [0x1f,0x24,0x03,0xd5]

// HINT: hint #32   // encoding: [0x1f,0x24,0x03,0xd5]
// HINT: hint #32   // encoding: [0x1f,0x24,0x03,0xd5]

hint #32

// CHECK: bti r     // encoding: [0x1f,0x24,0x03,0xd5]
// NOBTIE: bti      // encoding: [0x1f,0x24,0x03,0xd5]
// HINT: hint #32   // encoding: [0x1f,0x24,0x03,0xd5]
