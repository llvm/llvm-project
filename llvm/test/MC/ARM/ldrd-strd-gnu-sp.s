// PR19320
// RUN: not llvm-mc -triple=armv7a-linux-gnueabi -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=V7
// RUN:     llvm-mc -triple=armv8a-linux-gnueabi -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=V8
  .text

// This tries to use the GNU ldrd/strd alias to create an ldrd/strd instruction
// using the sp register. This is valid for V8, but not earlier architectures.

  .arm

// V7: error: invalid instruction
// V8: ldrd    r12, sp, [r0, #32]      @ encoding: [0xd0,0xc2,0xc0,0xe1]
        ldrd    r12, [r0, #32]

// V7: error: invalid instruction
// V8: strd    r12, sp, [r0, #32]      @ encoding: [0xf0,0xc2,0xc0,0xe1]
        strd    r12, [r0, #32]

  .thumb

// V7: error: invalid instruction
// V8: ldrd    r12, sp, [r0, #32]      @ encoding: [0xd0,0xe9,0x08,0xcd]
        ldrd    r12, [r0, #32]

// V7: error: invalid instruction
// V8: strd    r12, sp, [r0, #32]      @ encoding: [0xc0,0xe9,0x08,0xcd]
        strd    r12, [r0, #32]
