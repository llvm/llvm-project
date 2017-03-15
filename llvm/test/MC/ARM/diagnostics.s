@ RUN: not llvm-mc -triple=armv7-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS --check-prefix=CHECK-ERRORS-V7 < %t %s
@ RUN: not llvm-mc -triple=armv8 < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS --check-prefix=CHECK-ERRORS-V8 < %t %s

@ Check for various assembly diagnostic messages on invalid input.

@ 's' bit on an instruction that can't accept it.
        mlss r1, r2, r3, r4
@ CHECK-ERRORS: error: instruction 'mls' can not set flags,
@ CHECK-ERRORS: but 's' suffix specified


        @ Out of range shift immediate values.
        adc r1, r2, r3, lsl #invalid
        adc r4, r5, r6, lsl #-1
        adc r4, r5, r6, lsl #32
        adc r4, r5, r6, lsr #-1
        adc r4, r5, r6, lsr #33
        adc r4, r5, r6, asr #-1
        adc r4, r5, r6, asr #33
        adc r4, r5, r6, ror #-1
        adc r4, r5, r6, ror #32

@ CHECK-ERRORS: error: invalid immediate shift value
@ CHECK-ERRORS:         adc r1, r2, r3, lsl #invalid
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsl #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsl #32
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsr #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsr #33
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, asr #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, asr #33
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, ror #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, ror #32

        @ Out of range shift immediate values for load/store.
        str r1, [r2, r3, lsl #invalid]
        ldr r4, [r5], r6, lsl #-1
        pld r4, [r5, r6, lsl #32]
        str r4, [r5], r6, lsr #-1
        ldr r4, [r5, r6, lsr #33]
        pld r4, [r5, r6, asr #-1]
        str r4, [r5, r6, asr #33]
        ldr r4, [r5, r6, ror #-1]
        pld r4, [r5, r6, ror #32]
        pld r4, [r5, r6, rrx #0]

@ CHECK-ERRORS: error: shift amount must be an immediate
@ CHECK-ERRORS:         str r1, [r2, r3, lsl #invalid]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         ldr r4, [r5], r6, lsl #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         pld r4, [r5, r6, lsl #32]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         str r4, [r5], r6, lsr #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         ldr r4, [r5, r6, lsr #33]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         pld r4, [r5, r6, asr #-1]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         str r4, [r5, r6, asr #33]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         ldr r4, [r5, r6, ror #-1]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         pld r4, [r5, r6, ror #32]
@ CHECK-ERRORS: error: ']' expected
@ CHECK-ERRORS:         pld r4, [r5, r6, rrx #0]
        
        @ Out of range 16-bit immediate on BKPT
        bkpt #65536

@ CHECK-ERRORS: error: invalid operand for instruction

        @ Out of range immediates for v8 HLT instruction.
        hlt #65536
        hlt #-1
@CHECK-ERRORS: error: invalid operand for instruction
@CHECK-ERRORS:         hlt #65536
@CHECK-ERRORS:              ^
@CHECK-ERRORS: error: invalid operand for instruction
@CHECK-ERRORS:         hlt #-1
@CHECK-ERRORS:              ^

        @ Illegal condition code for v8 HLT instruction.
        hlteq #2
        hltlt #23
@CHECK-ERRORS: error: instruction 'hlt' is not predicable, but condition code specified
@CHECK-ERRORS:        hlteq #2
@CHECK-ERRORS:        ^
@CHECK-ERRORS: error: instruction 'hlt' is not predicable, but condition code specified
@CHECK-ERRORS:        hltlt #23
@CHECK-ERRORS:        ^

        @ Out of range 4 and 3 bit immediates on CDP[2]

        @ Out of range immediates for CDP/CDP2
        cdp  p7, #2, c1, c1, c1, #8
        cdp  p7, #1, c1, c1, c1, #8
        cdp2  p7, #2, c1, c1, c1, #8
        cdp2  p7, #1, c1, c1, c1, #8

@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction

        @ Out of range immediates for DBG
        dbg #-1
        dbg #16

@ CHECK-ERRORS: error: immediate operand must be in the range [0,15]
@ CHECK-ERRORS: error: immediate operand must be in the range [0,15]
@  Double-check that we're synced up with the right diagnostics.
@ CHECK-ERRORS: dbg #16

        @ Out of range immediate for MCR/MCR2/MCRR/MCRR2
        mcr  p7, #8, r5, c1, c1, #4
        mcr  p7, #2, r5, c1, c1, #8
        mcr2  p7, #8, r5, c1, c1, #4
        mcr2  p7, #1, r5, c1, c1, #8
        mcrr  p7, #16, r5, r4, c1
        mcrr2  p7, #16, r5, r4, c1
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: immediate operand must be in the range [0,15]
@ CHECK-ERRORS-V7: error: immediate operand must be in the range [0,15]
@ CHECK-ERRORS-V8: error: invalid operand for instruction

        @ p10 and p11 are reserved for NEON
        mcr p10, #2, r5, c1, c1, #4
        mcrr p11, #8, r5, r4, c1
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction

        @ Out of range immediate for MOV
        movw r9, 0x10000
@ CHECK-ERRORS: error: invalid operand for instruction

        @ Invalid 's' bit usage for MOVW
        movs r6, #0xffff
        movwseq r9, #0xffff
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: instruction 'movw' can not set flags, but 's' suffix specified

        @ Out of range immediate for MOVT
        movt r9, 0x10000
@ CHECK-ERRORS: error: invalid operand for instruction

        @ Out of range immediates for MRC/MRC2/MRRC/MRRC2
        mrc  p14, #8, r1, c1, c2, #4
        mrc  p14, #1, r1, c1, c2, #8
        mrc2  p14, #8, r1, c1, c2, #4
        mrc2  p14, #0, r1, c1, c2, #9
        mrrc  p7, #16, r5, r4, c1
        mrrc2  p7, #17, r5, r4, c1
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: immediate operand must be in the range [0,15]
@ CHECK-ERRORS-V7: error: immediate operand must be in the range [0,15]
@ CHECK-ERRORS-V8: error: invalid operand for instruction

        @ Shifter operand validation for PKH instructions.
        pkhbt r2, r2, r3, lsl #-1
        pkhbt r2, r2, r3, lsl #32
        pkhtb r2, r2, r3, asr #0
        pkhtb r2, r2, r3, asr #33
        pkhbt r2, r2, r3, asr #3
        pkhtb r2, r2, r3, lsl #3

@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS:         pkhbt r2, r2, r3, lsl #-1
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS:         pkhbt r2, r2, r3, lsl #32
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS:         pkhtb r2, r2, r3, asr #0
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS:         pkhtb r2, r2, r3, asr #33
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: lsl operand expected.
@ CHECK-ERRORS:         pkhbt r2, r2, r3, asr #3
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: error: asr operand expected.
@ CHECK-ERRORS:         pkhtb r2, r2, r3, lsl #3
@ CHECK-ERRORS:                           ^


        @ bad values for SETEND
        setendne be
        setend me
        setend 1

@ CHECK-ERRORS: error: instruction 'setend' is not predicable, but condition code specified
@ CHECK-ERRORS:         setendne be
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: 'be' or 'le' operand expected
@ CHECK-ERRORS:         setend me
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: error: 'be' or 'le' operand expected
@ CHECK-ERRORS:         setend 1
@ CHECK-ERRORS:                ^


        @ Out of range immediates and bad shift types for SSAT
	ssat	r8, #0, r10, lsl #8
	ssat	r8, #33, r10, lsl #8
	ssat	r8, #1, r10, lsl #-1
	ssat	r8, #1, r10, lsl #32
	ssat	r8, #1, r10, asr #0
	ssat	r8, #1, r10, asr #33
        ssat    r8, #1, r10, lsr #5
        ssat    r8, #1, r10, lsl fred
        ssat    r8, #1, r10, lsl #fred

@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: 	ssat	r8, #0, r10, lsl #8
@ CHECK-ERRORS: 	    	    ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: 	ssat	r8, #33, r10, lsl #8
@ CHECK-ERRORS: 	    	    ^
@ CHECK-ERRORS: error: 'lsr' shift amount must be in range [0,31]
@ CHECK-ERRORS: 	ssat	r8, #1, r10, lsl #-1
@ CHECK-ERRORS: 	    	                  ^
@ CHECK-ERRORS: error: 'lsr' shift amount must be in range [0,31]
@ CHECK-ERRORS: 	ssat	r8, #1, r10, lsl #32
@ CHECK-ERRORS: 	    	                  ^
@ CHECK-ERRORS: error: 'asr' shift amount must be in range [1,32]
@ CHECK-ERRORS: 	ssat	r8, #1, r10, asr #0
@ CHECK-ERRORS: 	    	                  ^
@ CHECK-ERRORS: error: 'asr' shift amount must be in range [1,32]
@ CHECK-ERRORS: 	ssat	r8, #1, r10, asr #33
@ CHECK-ERRORS: 	    	                  ^
@ CHECK-ERRORS: error: shift operator 'asr' or 'lsl' expected
@ CHECK-ERRORS:         ssat    r8, #1, r10, lsr #5
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: '#' expected
@ CHECK-ERRORS:         ssat    r8, #1, r10, lsl fred
@ CHECK-ERRORS:                                  ^
@ CHECK-ERRORS: error: shift amount must be an immediate
@ CHECK-ERRORS:         ssat    r8, #1, r10, lsl #fred
@ CHECK-ERRORS:                                   ^

        @ Out of range immediates for SSAT16
	ssat16	r2, #0, r7
	ssat16	r3, #17, r5

@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: 	ssat16	r2, #0, r7
@ CHECK-ERRORS: 	      	    ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: 	ssat16	r3, #17, r5
@ CHECK-ERRORS: 	      	    ^


        @ Out of order STM registers
        stmda sp!, {r5, r2}

@ CHECK-ERRORS: warning: register list not in ascending order
@ CHECK-ERRORS:         stmda     sp!, {r5, r2}
@ CHECK-ERRORS:                            ^


        @ Out of range immediate on SVC
        svc #0x1000000
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:   svc #0x1000000
@ CHECK-ERRORS:       ^


        @ Out of order Rt/Rt2 operands for ldrexd/strexd
        ldrexd  r4, r3, [r8]
        strexd  r6, r5, r3, [r8]

@ CHECK-ERRORS: error: destination operands must be sequential
@ CHECK-ERRORS:         ldrexd  r4, r3, [r8]
@ CHECK-ERRORS:                     ^
@ CHECK-ERRORS: error: source operands must be sequential
@ CHECK-ERRORS:         strexd  r6, r5, r3, [r8]
@ CHECK-ERRORS:                         ^

        @ Illegal rotate operators for extend instructions
        sxtb r8, r3, #8
        sxtb r8, r3, ror 24
        sxtb r8, r3, ror #8 -
        sxtab r3, r8, r3, ror #(fred - wilma)
        sxtab r7, r8, r3, ror #25
        sxtah r9, r3, r3, ror #-8
        sxtb16ge r2, r3, lsr #24

@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         sxtb r8, r3, #8
@ CHECK-ERRORS:                      ^
@ CHECK-ERRORS: error: '#' expected
@ CHECK-ERRORS:         sxtb r8, r3, ror 24
@ CHECK-ERRORS:                          ^
@ CHECK-ERRORS: error: unknown token in expression
@ CHECK-ERRORS:         sxtb r8, r3, ror #8 -
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: malformed rotate expression
@ CHECK-ERRORS:         sxtb r8, r3, ror #8 -
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: error: rotate amount must be an immediate
@ CHECK-ERRORS:         sxtab r3, r8, r3, ror #(fred - wilma)
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: 'ror' rotate amount must be 8, 16, or 24
@ CHECK-ERRORS:         sxtab r7, r8, r3, ror #25
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: 'ror' rotate amount must be 8, 16, or 24
@ CHECK-ERRORS:         sxtah r9, r3, r3, ror #-8
@ CHECK-ERRORS:                                ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         sxtb16ge r2, r3, lsr #24
@ CHECK-ERRORS:                          ^

        @ Out of range width for SBFX/UBFX
        sbfx r4, r5, #31, #2
        ubfxgt r4, r5, #16, #17

@ CHECK-ERRORS: error: bitfield width must be in range [1,32-lsb]
@ CHECK-ERRORS:         sbfx r4, r5, #31, #2
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: error: bitfield width must be in range [1,32-lsb]
@ CHECK-ERRORS:         ubfxgt r4, r5, #16, #17
@ CHECK-ERRORS:                             ^

        @ Using pc for SBFX/UBFX
        sbfx pc, r2, #1, #3
        sbfx sp, pc, #4, #5
        ubfx pc, r0, #0, #31
        ubfx r14, pc, #1, #2
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         sbfx pc, r2, #1, #3
@ CHECK-ERRORS:              ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         sbfx sp, pc, #4, #5
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         ubfx pc, r0, #0, #31
@ CHECK-ERRORS:              ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         ubfx r14, pc, #1, #2
@ CHECK-ERRORS:                   ^

        @ Out of order Rt/Rt2 operands for ldrd
        ldrd  r4, r3, [r8]
        ldrd  r4, r3, [r8, #8]!
        ldrd  r4, r3, [r8], #8
@ CHECK-ERRORS: error: destination operands must be sequential
@ CHECK-ERRORS:         ldrd  r4, r3, [r8]
@ CHECK-ERRORS:                   ^
@ CHECK-ERRORS: error: destination operands must be sequential
@ CHECK-ERRORS:         ldrd  r4, r3, [r8, #8]!
@ CHECK-ERRORS:                   ^
@ CHECK-ERRORS: error: destination operands must be sequential
@ CHECK-ERRORS:         ldrd  r4, r3, [r8], #8
@ CHECK-ERRORS:                   ^


        @ Bad register lists for VFP.
        vpush {s0, s3}
@ CHECK-ERRORS: error: non-contiguous register range
@ CHECK-ERRORS:         vpush {s0, s3}
@ CHECK-ERRORS:                    ^

        @ Out of range coprocessor option immediate.
        ldc2 p2, c8, [r1], { 256 }
        ldc2 p2, c8, [r1], { -1 }

@ CHECK-ERRORS-V7: error: coprocessor option must be an immediate in range [0, 255]
@ CHECK-ERRORS-V7:         ldc2 p2, c8, [r1], { 256 }
@ CHECK-ERRORS-V7:                              ^
@ CHECK-ERRORS-V8: error: register expected
@ CHECK-ERRORS-V7: error: coprocessor option must be an immediate in range [0, 255]
@ CHECK-ERRORS-V7:         ldc2 p2, c8, [r1], { -1 }
@ CHECK-ERRORS-V7:                              ^
@ CHECK-ERRORS-V8: error: register expected

        @ Bad CPS instruction format.
        cps f,#1
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         cps f,#1
@ CHECK-ERRORS:               ^

        @ Bad operands for msr
        msr #0, #0
        msr foo, #0
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         msr #0, #0
@ CHECK-ERRORS:             ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         msr foo, #0
@ CHECK-ERRORS:             ^

        isb #-1
        isb #16
@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS: error: immediate value out of range

        nop.n
@ CHECK-ERRORS: error: instruction with .n (narrow) qualifier not allowed in arm mode

	dmbeq #5
	dsble #15
	isblo #7
@ CHECK-ERRORS: error: instruction 'dmb' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'dsb' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'isb' is not predicable, but condition code specified

	dmblt
	dsbne
	isbeq
@ CHECK-ERRORS: error: instruction 'dmb' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'dsb' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'isb' is not predicable, but condition code specified

        mcr2le  p7, #1, r5, c1, c1, #4
        mcrr2ne p7, #15, r5, r4, c1
        mrc2lo  p14, #0, r1, c1, c2, #4
        mrrc2lo  p7, #1, r5, r4, c1
        cdp2hi   p10, #0, c6, c12, c0, #7
@ CHECK-ERRORS: error: instruction 'mcr2' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'mcrr2' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'mrc2' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'mrrc2' is not predicable, but condition code specified
@ CHECK-ERRORS: error: instruction 'cdp2' is not predicable, but condition code specified

        bkpteq #7
@ CHECK-ERRORS: error: instruction 'bkpt' is not predicable, but condition code specified

        ldm r2!, {r2, r3}
        ldmdb r2!, {r2, r3}
        ldmda r2!, {r2, r3}
        popeq {sp}
@ CHECK-ERRORS: error: writeback register not allowed in register list
@ CHECK-ERRORS: error: writeback register not allowed in register list
@ CHECK-ERRORS: error: writeback register not allowed in register list
@ CHECK-ERRORS: error: writeback register not allowed in register list

        vrintz.f32.f32 s0, s1
        vrintr.f32 s0, s1
        vrintx.f64.f64 d2, d5
        vrintz.f64 d10, d9
        vrinta.f32.f32 s6, s7
        vrintn.f32 s8, s9
        vrintp.f64.f64 d10, d11
        vrintm.f64 d12, d13
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8
@ CHECK-ERRORS-V7: error: instruction requires: FPARMv8

        stm sp!, {r0, pc}^
        ldm sp!, {r0}^
@ CHECK-ERRORS: error: system STM cannot have writeback register
@ CHECK-ERRORS: error: writeback register only allowed on system LDM if PC in register-list

foo2:
        mov r0, foo2
        movw r0, foo2
        movt r0, foo2
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                 ^
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                  ^

        str r0, [r0, #4]!
        str r0, [r0, r1]!
        str r0, [r0], #4
        str r0, [r0], r1
        strh r0, [r0, #2]!
        strh r0, [r0, r1]!
        strh r0, [r0], #2
        strh r0, [r0], r1
        strb r0, [r0, #1]!
        strb r0, [r0, r1]!
        strb r0, [r0], #1
        strb r0, [r0], r1
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: str r0, [r0, #4]!
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: str r0, [r0, r1]!
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: str r0, [r0], #4
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: str r0, [r0], r1
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strh r0, [r0, #2]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strh r0, [r0, r1]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strh r0, [r0], #2
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strh r0, [r0], r1
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strb r0, [r0, #1]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strb r0, [r0, r1]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strb r0, [r0], #1
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: source register and base register can't be identical
@ CHECK-ERRORS: strb r0, [r0], r1
@ CHECK-ERRORS:          ^

        ldr r0, [r0, #4]!
        ldr r0, [r0, r1]!
        ldr r0, [r0], #4
        ldr r0, [r0], r1
        ldrh r0, [r0, #2]!
        ldrh r0, [r0, r1]!
        ldrh r0, [r0], #2
        ldrh r0, [r0], r1
        ldrsh r0, [r0, #2]!
        ldrsh r0, [r0, r1]!
        ldrsh r0, [r0], #2
        ldrsh r0, [r0], r1
        ldrb r0, [r0, #1]!
        ldrb r0, [r0, r1]!
        ldrb r0, [r0], #1
        ldrb r0, [r0], r1
        ldrsb r0, [r0, #1]!
        ldrsb r0, [r0, r1]!
        ldrsb r0, [r0], #1
        ldrsb r0, [r0], r1
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldr r0, [r0, #4]!
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldr r0, [r0, r1]!
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldr r0, [r0], #4
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldr r0, [r0], r1
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrh r0, [r0, #2]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrh r0, [r0, r1]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrh r0, [r0], #2
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrh r0, [r0], r1
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsh r0, [r0, #2]!
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsh r0, [r0, r1]!
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsh r0, [r0], #2
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsh r0, [r0], r1
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrb r0, [r0, #1]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrb r0, [r0, r1]!
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrb r0, [r0], #1
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrb r0, [r0], r1
@ CHECK-ERRORS:          ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsb r0, [r0, #1]!
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsb r0, [r0, r1]!
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsb r0, [r0], #1
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: destination register and base register can't be identical
@ CHECK-ERRORS: ldrsb r0, [r0], r1
@ CHECK-ERRORS:           ^

        @ Out of range modified immediate values
        mov  r5, #-256, #6
        mov  r6, #42, #7
        mvn  r5, #256, #6
        mvn  r6, #42, #298
        cmp  r5, #65535, #6
        cmp  r6, #42, #31
        cmn  r5, #-1, #6
        cmn  r6, #42, #32
	msr  APSR_nzcvq, #-128, #2
	msr  apsr_nzcvqg, #0, #1
        adc  r7, r8, #-256, #2
        adc  r7, r8, #128, #1
        sbc  r7, r8, #-256, #2
        sbc  r7, r8, #128, #1
        add  r7, r8, #-2149, #0
        add  r7, r8, #100, #1
        sub  r7, r8, #-2149, #0
        sub  r7, r8, #100, #1
        and  r7, r8, #-2149, #0
        and  r7, r8, #100, #1
        orr  r7, r8, #-2149, #0
        orr  r7, r8, #100, #1
        eor  r7, r8, #-2149, #0
        eor  r7, r8, #100, #1
        bic  r7, r8, #-2149, #0
        bic  r7, r8, #100, #1
        rsb  r7, r8, #-2149, #0
        rsb  r7, r8, #100, #1
        adds r7, r8, #-2149, #0
        adds r7, r8, #100, #1
        subs r7, r8, #-2149, #0
        subs r7, r8, #100, #1
        rsbs r7, r8, #-2149, #0
        rsbs r7, r8, #100, #1
        rsc r7, r8, #-2149, #0
        rsc r7, r8, #100, #1
        TST r7, #-2149, #0
        TST r7, #100, #1
        TEQ r7, #-2149, #0
        TEQ r7, #100, #1
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
@ CHECK-ERRORS: error: immediate operand must a number in the range [0, 255]
@ CHECK-ERRORS: error: immediate operand must an even number in the range [0, 30]
