@ RUN: not llvm-mc -triple=thumbv6-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s
@ RUN: not llvm-mc -triple=thumbv5-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS-V5 < %t %s

@ Check for various assembly diagnostic messages on invalid input.

@ ADD instruction w/o 'S' suffix.
        add r1, r2, r3
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-ERRORS:         add r1, r2, r3
@ CHECK-ERRORS:         ^

@ Instructions which require v6+ for both registers to be low regs.
        add r2, r3
        mov r2, r3
@ CHECK-ERRORS: error: instruction variant requires Thumb2
@ CHECK-ERRORS:         add r2, r3
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS-V5: error: instruction variant requires ARMv6 or later
@ CHECK-ERRORS-V5:         mov r2, r3
@ CHECK-ERRORS-V5:         ^


@ Out of range immediates for ASR instruction.
        asrs r2, r3, #33
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         asrs r2, r3, #33
@ CHECK-ERRORS:                      ^

@ Out of range immediates for BKPT instruction.
        bkpt #256
        bkpt #-1
error: invalid operand for instruction
        bkpt #256
             ^
error: invalid operand for instruction
        bkpt #-1
             ^

@ Invalid writeback and register lists for LDM
        ldm r2!, {r5, r8}
        ldm r2, {r5, r7}
        ldm r2!, {r2, r3, r4}
@ CHECK-ERRORS: error: registers must be in range r0-r7
@ CHECK-ERRORS:         ldm r2!, {r5, r8}
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: error: writeback operator '!' expected
@ CHECK-ERRORS:         ldm r2, {r5, r7}
@ CHECK-ERRORS:             ^
@ CHECK-ERRORS: error: writeback operator '!' not allowed when base register in register list
@ CHECK-ERRORS:         ldm r2!, {r2, r3, r4}
@ CHECK-ERRORS:               ^


@ Invalid writeback and register lists for PUSH/POP
        pop {r1, r2, r10}
        push {r8, r9}
@ CHECK-ERRORS: error: registers must be in range r0-r7 or pc
@ CHECK-ERRORS:         pop {r1, r2, r10}
@ CHECK-ERRORS:             ^
@ CHECK-ERRORS: error: registers must be in range r0-r7 or lr
@ CHECK-ERRORS:         push {r8, r9}
@ CHECK-ERRORS:              ^


@ Invalid writeback and register lists for STM
        stm r1, {r2, r6}
        stm r1!, {r2, r9}
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         stm r1, {r2, r6}
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: registers must be in range r0-r7
@ CHECK-ERRORS:         stm r1!, {r2, r9}
@ CHECK-ERRORS:                  ^

@ Out of range immediates for LSL instruction.
        lsls r4, r5, #-1
        lsls r4, r5, #32
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         lsls r4, r5, #-1
@ CHECK-ERRORS:                      ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         lsls r4, r5, #32
@ CHECK-ERRORS:                      ^

@ Mismatched source/destination operands for MUL instruction.
        muls r1, r2, r3
@ CHECK-ERRORS: error: destination register must match source register
@ CHECK-ERRORS:         muls r1, r2, r3
@ CHECK-ERRORS:              ^


@ Out of range immediates for STR instruction.
        str r2, [r7, #-1]
        str r5, [r1, #3]
        str r3, [r7, #128]
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         str r2, [r7, #-1]
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         str r5, [r1, #3]
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         str r3, [r7, #128]
@ CHECK-ERRORS:         ^

@ Out of range immediate for SVC instruction.
        svc #-1
        svc #256
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         svc #-1
@ CHECK-ERRORS:             ^
@ CHECK-ERRORS: error: instruction requires: arm-mode
@ CHECK-ERRORS:         svc #256
@ CHECK-ERRORS:         ^


@ Out of range immediate for ADD SP instructions
        add sp, #-1
        add sp, #3
        add sp, sp, #512
        add r2, sp, #1024
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         add sp, #-1
@ CHECK-ERRORS:                 ^
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         add sp, #3
@ CHECK-ERRORS:                 ^
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS:         add sp, sp, #512
@ CHECK-ERRORS:                     ^
@ CHECK-ERRORS: error: instruction requires: arm-mode
@ CHECK-ERRORS:         add r2, sp, #1024
@ CHECK-ERRORS:         ^

        add r2, sp, ip
@ CHECK-ERRORS: error: source register must be the same as destination
@ CHECK-ERRORS:         add r2, sp, ip
@ CHECK-ERRORS:                     ^
 
@------------------------------------------------------------------------------
@ WFE/WFI/YIELD - are not supported pre v6T2
@------------------------------------------------------------------------------
        wfe
        wfi
        yield

@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS: wfe
@ CHECK-ERRORS: ^
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS: wfi
@ CHECK-ERRORS: ^
@ CHECK-ERRORS: error: instruction requires: thumb2
@ CHECK-ERRORS: yield
@ CHECK-ERRORS: ^

