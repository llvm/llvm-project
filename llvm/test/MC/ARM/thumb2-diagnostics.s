@ RUN: not llvm-mc -triple=thumbv7-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS --check-prefix=CHECK-ERRORS-V7 < %t %s

@ RUN: not llvm-mc -triple=thumbv8-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS --check-prefix=CHECK-ERRORS-V8 < %t %s

@ Ill-formed IT block instructions.
        itet eq
        addle r0, r1, r2
        nop
        it le
        iteeee gt
        ittfe le
        nopeq

@ CHECK-ERRORS: error: incorrect condition in IT block; got 'le', but expected 'eq'
@ CHECK-ERRORS:         addle r0, r1, r2
@ CHECK-ERRORS:            ^
@ CHECK-ERRORS: error: incorrect condition in IT block; got 'al', but expected 'ne'
@ CHECK-ERRORS:         nop
@ CHECK-ERRORS:            ^
@ CHECK-ERRORS: error: instructions in IT block must be predicable
@ CHECK-ERRORS:         it le
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: too many conditions on IT instruction
@ CHECK-ERRORS:         iteeee gt
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: illegal IT block condition mask 'tfe'
@ CHECK-ERRORS:         ittfe le
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: predicated instructions must be in IT block
@ CHECK-ERRORS:         nopeq
@ CHECK-ERRORS:         ^

        @ Out of range immediates for MRC/MRC2/MRRC/MRRC2
        mrc  p14, #8, r1, c1, c2, #4
        mrc  p14, #1, r1, c1, c2, #8
        mrc2  p14, #8, r1, c1, c2, #4
        mrc2  p14, #0, r1, c1, c2, #9
        mrrc  p14, #16, r5, r4, c1
        mrrc2  p14, #17, r5, r4, c1
@ CHECK-ERRORS: operand must be an immediate in the range [0,7]
@ CHECK-ERRORS: operand must be an immediate in the range [0,7]
@ CHECK-ERRORS-V7: operand must be an immediate in the range [0,7]
@ CHECK-ERRORS-V7: operand must be an immediate in the range [0,7]
@ CHECK-ERRORS-V8: invalid instruction
@ CHECK-ERRORS-V8: too many operands for instruction
@ CHECK-ERRORS: operand must be an immediate in the range [0,15]
@ CHECK-ERRORS-V7: operand must be an immediate in the range [0,15]
@ CHECK-ERRORS-V8: invalid instruction

        @ Out of range immediate for ROR.
        @ (Assembling this instruction to "mov r1, r1" might also be OK.)
        ror r1, r1, #0
@ CHECK-ERRORS: invalid instruction
@ CHECK-ERRORS: operand must be an immediate in the range [1,31]

        isb  #-1
        isb  #16
@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS: error: immediate value out of range

        itt eq
        bkpteq #1
@ CHECK-ERRORS: error: instruction 'bkpt' is not predicable, but condition code specified

        nopeq
        nopeq

@ out of range operands for Thumb2 targets

        beq.w  #-1048578
        bne.w  #1048576
        blt.w  #1013411
        b.w    #-16777218
        b.w    #16777216
        b.w    #1592313

@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range

foo2:
        movw r0, foo2
        movt r0, foo2
        movt r0, #0x10000
        movt r0, #0x10000
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: error: operand must be an immediate in the range [0,0xffff] or a relocatable expression
@ CHECK-ERRORS: error: operand must be an immediate in the range [0,0xffff] or a relocatable expression

        and sp, r1, #80008000
        and pc, r1, #80008000
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-ERRORS: error: invalid instruction

        ssat r0, #1, r0, asr #32
        usat r0, #1, r0, asr #32
@ CHECK-ERRORS: error: 'asr #32' shift amount not allowed in Thumb mode
@ CHECK-ERRORS: error: 'asr #32' shift amount not allowed in Thumb mode

        @ PC is not valid as shifted-rGPR
        sbc.w r2, r7, pc, lsr #16
        and.w r2, r7, pc, lsr #16
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: invalid operand for instruction
@ CHECK-ERRORS-V7: note: operand must be a register in range [r0, r12] or r14
@ CHECK-ERRORS-V8: note: operand must be a register in range [r0, r14]
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: invalid operand for instruction
@ CHECK-ERRORS-V7: note: operand must be a register in range [r0, r12] or r14
@ CHECK-ERRORS-V8: note: operand must be a register in range [r0, r14]


        @ PC is not valid as base of load
        ldr r0, [pc, r0]
        ldrb r1, [pc, r2]
        ldrh r3, [pc, r3]
        pld r4, [pc, r5]
        str r6, [pc, r7]
        strb r7 [pc, r8]
        strh r9, [pc, r10]
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: invalid operand for instruction
@ CHECK-ERRORS: note: instruction requires: arm-mode
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: invalid operand for instruction
@ CHECK-ERRORS: note: instruction requires: arm-mode
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: instruction requires: arm-mode
@ CHECK-ERRORS: note: invalid operand for instruction
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: invalid operand for instruction
@ CHECK-ERRORS: note: instruction requires: arm-mode
@ CHECK-ERRORS: error: immediate value expected for vector index
@ CHECK-ERRORS: error: invalid instruction, any one of the following would fix this:
@ CHECK-ERRORS: note: instruction requires: arm-mode
@ CHECK-ERRORS: note: invalid operand for instruction

        @ SWP(B) is an ARM-only instruction
        swp  r0, r1, [r2]
        swpb r3, r4, [r5]
@ CHECK-ERRORS-V7: error: instruction requires: arm-mode
@ CHECK-ERRORS-V7: error: instruction requires: arm-mode
@ CHECK-ERRORS-V8: error: invalid instruction
@ CHECK-ERRORS-V8: error: invalid instruction

        @ Generic error for too few operands
        adds
        adds r0
@ CHECK-ERRORS: error: too few operands for instruction
@ CHECK-ERRORS: error: too few operands for instruction

        tst sp, #3
        tst sp, r5
        tst sp, r5, lsl #3
@ CHECK-ERRORS-V7: error: operand must be a register in range [r0, r12] or r14
@ CHECK-ERRORS-V7: operand must be a register in range [r0, r12] or r14
@ CHECK-ERRORS-V7: operand must be a register in range [r0, r12] or r14

        teq sp, #5
        teq sp, r7
        teq sp, r9, lsl #2
@ CHECK-ERRORS-V7: error: operand must be a register in range [r0, r12] or r14
@ CHECK-ERRORS-V7: operand must be a register in range [r0, r12] or r14
@ CHECK-ERRORS-V7: operand must be a register in range [r0, r12] or r14

        tbb [r0, sp]
        @ v8 allows rm = sp
@ CHECK-ERRORS-V7: error: instruction variant requires ARMv8 or later
        tbb [r0, pc]
        @ rm = pc is always unpredictable
@ CHECK-ERRORS: error: invalid operand for instruction
        tbb [sp, r0]
        @ v8 allows rn = sp
@ CHECK-ERRORS-V7: error: instruction variant requires ARMv8 or later
        @ rn = pc is allowed so not included here

        tbh [r0, sp, lsl #1]
        @ v8 allows rm = sp
@ CHECK-ERRORS-V7: error: instruction variant requires ARMv8 or later
        tbh [r0, pc, lsl #1]
        @ rm = pc is always unpredictable
@ CHECK-ERRORS: error: invalid operand for instruction
        tbh [sp, r0, lsl #1]
        @ v8 allows rn = sp
@ CHECK-ERRORS-V7: error: instruction variant requires ARMv8 or later
        @ rn=pc is allowed so not included here
