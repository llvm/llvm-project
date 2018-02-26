// RUN: not llvm-mc -triple=thumbv7 -show-encoding < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-NONARM --check-prefix=CHECK-THUMBV7 %s
// RUN: not llvm-mc -triple=thumbv8 -show-encoding < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-NONARM --check-prefix=CHECK-THUMBV8 %s
// RUN: llvm-mc -triple=armv7 -show-encoding < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-ARM %s

        // lsl #0 is actually mov, so here we check that it behaves the same as
        // mov with regards to the permitted registers

        // Using PC is invalid in thumb
        lsl pc, r0, #0
        lsl r0, pc, #0
        lsl pc, pc, #0
        lsls pc, r0, #0
        lsls r0, pc, #0
        lsls pc, pc, #0

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: lsl pc, r0, #0
// CHECK-NONARM: note: instruction requires: arm-mode
// CHECK-NONARM: note: operand must be a register in range [r0, r14]

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: lsl r0, pc, #0
// CHECK-NONARM: note: instruction requires: arm-mode
// CHECK-NONARM: note: operand must be a register in range [r0, r14]

// CHECK-NONARM: error: instruction requires: arm-mode
// CHECK-NONARM-NEXT: lsl pc, pc, #0

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: lsls pc, r0, #0
// CHECK-NONARM: note: instruction requires: arm-mode
// CHECK-NONARM: note: operand must be a register in range [r0, r14]

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: lsls r0, pc, #0
// CHECK-NONARM: note: instruction requires: arm-mode
// CHECK-NONARM: note: operand must be a register in range [r0, r14]

// CHECK-NONARM: error: instruction requires: arm-mode
// CHECK-NONARM-NEXT: lsls pc, pc, #0

// CHECK-ARM: mov pc, r0                @ encoding: [0x00,0xf0,0xa0,0xe1]
// CHECK-ARM: mov r0, pc                @ encoding: [0x0f,0x00,0xa0,0xe1]
// CHECK-ARM: mov pc, pc                @ encoding: [0x0f,0xf0,0xa0,0xe1]
// CHECK-ARM: movs pc, r0               @ encoding: [0x00,0xf0,0xb0,0xe1]
// CHECK-ARM: movs r0, pc               @ encoding: [0x0f,0x00,0xb0,0xe1]
// CHECK-ARM: movs pc, pc               @ encoding: [0x0f,0xf0,0xb0,0xe1]

        mov pc, r0, lsl #0
        mov r0, pc, lsl #0
        mov pc, pc, lsl #0
        movs pc, r0, lsl #0
        movs r0, pc, lsl #0
        movs pc, pc, lsl #0

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: mov pc, r0, lsl #0
// CHECK-NONARM: note: operand must be a register in range [r0, r15]
// CHECK-THUMBV7: note: operand must be a register in range [r0, r12] or r14
// CHECK-THUMBV8: note: operand must be a register in range [r0, r14]

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: mov r0, pc, lsl #0
// CHECK-NONARM: note: operand must be a register in range [r0, r15]
// CHECK-NONARM: note: invalid operand for instruction
// CHECK-NONARM: note: invalid operand for instruction
// CHECK-NONARM: note: operand must be an immediate in the range [256,65535]

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: mov pc, pc, lsl #0
// CHECK-NONARM: note: operand must be a register in range [r0, r15]
// CHECK-THUMBV7: note: operand must be a register in range [r0, r12] or r14
// CHECK-THUMBV8: note: operand must be a register in range [r0, r14]

// CHECK-THUMBV7: error: operand must be a register in range [r0, r12] or r14
// CHECK-THUMBV8: error: operand must be a register in range [r0, r14]
// CHECK-NONARM-NEXT: movs pc, r0, lsl #0

// CHECK-NONARM: error: invalid instruction, any one of the following would fix this:
// CHECK-NONARM-NEXT: movs r0, pc, lsl #0
// CHECK-NONARM: note: operand must be a register in range [r0, r14]
// CHECK-NONARM: note: invalid operand for instruction
// CHECK-NONARM: note: invalid operand for instruction

// CHECK-THUMBV7: error: operand must be a register in range [r0, r12] or r14
// CHECK-THUMBV8: error: operand must be a register in range [r0, r14]
// CHECK-NONARM-NEXT: movs pc, pc, lsl #0

// CHECK-ARM: mov pc, r0                @ encoding: [0x00,0xf0,0xa0,0xe1]
// CHECK-ARM: mov r0, pc                @ encoding: [0x0f,0x00,0xa0,0xe1]
// CHECK-ARM: mov pc, pc                @ encoding: [0x0f,0xf0,0xa0,0xe1]
// CHECK-ARM: movs pc, r0               @ encoding: [0x00,0xf0,0xb0,0xe1]
// CHECK-ARM: movs r0, pc               @ encoding: [0x0f,0x00,0xb0,0xe1]
// CHECK-ARM: movs pc, pc               @ encoding: [0x0f,0xf0,0xb0,0xe1]

        // Using SP is invalid before ARMv8 in thumb unless non-flags-setting
        // and one of the source and destination is not SP
        lsl sp, sp, #0
        lsls sp, sp, #0
        lsls r0, sp, #0
        lsls sp, r0, #0

// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: lsl sp, sp, #0
// CHECK-THUMBV7: instruction requires: arm-mode
// CHECK-THUMBV7: instruction variant requires ARMv8 or later
// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: lsls sp, sp, #0
// CHECK-THUMBV7: instruction requires: arm-mode
// CHECK-THUMBV7: instruction variant requires ARMv8 or later
// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: lsls r0, sp, #0
// CHECK-THUMBV7: instruction requires: arm-mode
// CHECK-THUMBV7: instruction variant requires ARMv8 or later
// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: lsls sp, r0, #0
// CHECK-THUMBV7: instruction requires: arm-mode
// CHECK-THUMBV7: instruction variant requires ARMv8 or later

// CHECK-ARM: mov sp, sp                @ encoding: [0x0d,0xd0,0xa0,0xe1]
// CHECK-ARM: movs sp, sp               @ encoding: [0x0d,0xd0,0xb0,0xe1]
// CHECK-ARM: movs r0, sp               @ encoding: [0x0d,0x00,0xb0,0xe1]
// CHECK-ARM: movs sp, r0               @ encoding: [0x00,0xd0,0xb0,0xe1]

        mov sp, sp, lsl #0
        movs sp, sp, lsl #0
        movs r0, sp, lsl #0
        movs sp, r0, lsl #0

// FIXME: We should consistently have the "requires ARMv8" error here
// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: mov sp, sp, lsl #0
// CHECK-THUMBV7: note: operand must be a register in range [r0, r15]
// CHECK-THUMBV7: note: operand must be a register in range [r0, r12] or r14

// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: movs sp, sp, lsl #0
// CHECK-THUMBV7: note: operand must be a register in range [r0, r14]
// CHECK-THUMBV7: note: operand must be a register in range [r0, r12] or r14

// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: movs r0, sp, lsl #0
// CHECK-THUMBV7: note: operand must be a register in range [r0, r14]
// CHECK-THUMBV7: note: invalid operand for instruction
// CHECK-THUMBV7: note: instruction variant requires ARMv8 or later

// CHECK-THUMBV7: error: invalid instruction, any one of the following would fix this:
// CHECK-THUMBV7-NEXT: movs sp, r0, lsl #0
// CHECK-THUMBV7: note: operand must be a register in range [r0, r14]
// CHECK-THUMBV7: note: operand must be a register in range [r0, r12] or r14

// CHECK-ARM: mov sp, sp                @ encoding: [0x0d,0xd0,0xa0,0xe1]
// CHECK-ARM: movs sp, sp               @ encoding: [0x0d,0xd0,0xb0,0xe1]
// CHECK-ARM: movs r0, sp               @ encoding: [0x0d,0x00,0xb0,0xe1]
// CHECK-ARM: movs sp, r0               @ encoding: [0x00,0xd0,0xb0,0xe1]
