// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+bti   < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-bti   < %s 2>&1 | FileCheck %s --check-prefix=NOBTI

bti
bti c
bti j
bti jc

// CHECK: bti      // encoding: [0x1f,0x24,0x03,0xd5]
// CHECK: bti c    // encoding: [0x5f,0x24,0x03,0xd5]
// CHECK: bti j    // encoding: [0x9f,0x24,0x03,0xd5]
// CHECK: bti jc   // encoding: [0xdf,0x24,0x03,0xd5]

// NOBTI:      instruction requires: bti
// NOBTI-NEXT: bti
// NOBTI:      instruction requires: bti
// NOBTI-NEXT: bti
// NOBTI:      instruction requires: bti
// NOBTI-NEXT: bti
// NOBTI:      instruction requires: bti
// NOBTI-NEXT: bti

hint #32
hint #34
hint #36
hint #38

// CHECK: bti      // encoding: [0x1f,0x24,0x03,0xd5]
// CHECK: bti c    // encoding: [0x5f,0x24,0x03,0xd5]
// CHECK: bti j    // encoding: [0x9f,0x24,0x03,0xd5]
// CHECK: bti jc   // encoding: [0xdf,0x24,0x03,0xd5]

// NOBTI: hint #32 // encoding: [0x1f,0x24,0x03,0xd5]
// NOBTI: hint #34 // encoding: [0x5f,0x24,0x03,0xd5]
// NOBTI: hint #36 // encoding: [0x9f,0x24,0x03,0xd5]
// NOBTI: hint #38 // encoding: [0xdf,0x24,0x03,0xd5]
