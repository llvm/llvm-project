// RUN: llvm-mc -triple thumbv7 -show-encoding < %s | FileCheck %s

// Test each of the Thumb1 data-processing instructions
// The assembly syntax for these instructions allows an optional Rd register
//   OP{S}{<c>}{<q>}  {<Rd>,} <Rn>, <Rm>
// Assemblers should chose the narrow thumb encoding when possible, i.e.
//   - Rd == Rn 
//   - Rd, Rn and Rm are < r8
// In addition, some operations are commutative, allowing the transformation
// when:
//   - Rd == Rn || Rd == Rm
//   - Rd, Rn and Rm are < r8

// ADD immediate (not SP) A8.8.4
    ADDS     r0, r0, #5          // T1
// CHECK: adds  r0, r0, #5          @ encoding: [0x40,0x1d]
    ADDS     r1, r1, #8          // T2
// CHECK: adds  r1, #8              @ encoding: [0x08,0x31]
    ADDS.W   r1, r1, #8          // .w => T3
// CHECK: adds.w r1, r1, #8         @ encoding: [0x11,0xf1,0x08,0x01]
    ADDS     r8, r8, #8          // T3
// CHECK: adds.w r8, r8, #8         @ encoding: [0x18,0xf1,0x08,0x08]

    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDEQ    r0, r0, #5          // T1
// CHECK: addeq r0, r0, #5          @ encoding: [0x40,0x1d]
    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDEQ    r1, r1, #8          // T2
// CHECK: addeq r1, #8              @ encoding: [0x08,0x31]

    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDSEQ   r0, r0, #5          // T3
// CHECK: addseq.w r0, r0, #5       @ encoding: [0x10,0xf1,0x05,0x00]
    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDSEQ   r1, r1, #8          // T3
// CHECK: addseq.w r1, r1, #8       @ encoding: [0x11,0xf1,0x08,0x01]

// ADD register (not SP) A8.8.6 (commutative)
    ADDS     r0, r2, r1          // ADDS has T1 narrow 3 operand
// CHECK: adds  r0, r2, r1          @ encoding: [0x50,0x18]
    ADDS     r2, r2, r1          // ADDS has T1 narrow 3 operand
// CHECK: adds  r2, r2, r1          @ encoding: [0x52,0x18]
    ADD      r3, r1, r3          // T2
// CHECK: add  r3, r1               @ encoding: [0x0b,0x44]

    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDEQ    r0, r2, r1          // (In IT) ADD has T1 narrow 3 operand
// CHECK: addeq r0, r2, r1          @ encoding: [0x50,0x18]
    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDEQ    r2, r2, r1          // (In IT) ADD has T1 narrow 3 operand
// CHECK: addeq r2, r2, r1          @ encoding: [0x52,0x18]

    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDSEQ   r0, r2, r1          // T3
// CHECK: addseq.w r0, r2, r1       @ encoding: [0x12,0xeb,0x01,0x00]
    IT EQ
// CHECK: it eq                     @ encoding: [0x08,0xbf]
    ADDSEQ   r2, r2, r1          // T3
// CHECK: addseq.w r2, r2, r1       @ encoding: [0x12,0xeb,0x01,0x02]

    ADD      r3, r3, r1          // T2
// CHECK: add  r3, r1               @ encoding: [0x0b,0x44]
    ADD      r4, r4, pc          // T2
// CHECK: add  r4, pc               @ encoding: [0x7c,0x44]
    ADD      r4, pc, r4          // T2
// CHECK: add  r4, pc               @ encoding: [0x7c,0x44]
    ADD      pc, pc, r2          // T2
// CHECK: add  pc, r2               @ encoding: [0x97,0x44]
    ADD      pc, r2, pc          // T2
// CHECK: add  pc, r2               @ encoding: [0x97,0x44]
    ADD      pc, pc, sp          // T2
// CHECK: add   pc, sp              @ encoding: [0xef,0x44]
    ADD      pc, sp, pc          // T2
// CHECK: add   pc, sp, pc          @ encoding: [0xef,0x44]

// ADD (SP plus immediate) A8.8.9
    ADD      sp, sp, #20         // T2
// FIXME: ARMARM says 'add   sp, sp, #20'
// CHECK: add   sp, #20             @ encoding: [0x05,0xb0]
    ADD      sp, sp, #508        // T2
// CHECK: add   sp, #508            @ encoding: [0x7f,0xb0]
    ADD      sp, sp, #512        // T3
// CHECK: add.w sp, sp, #512        @ encoding: [0x0d,0xf5,0x00,0x7d]

// ADD (SP plus register) A8.8.10 (commutative)
    ADD      r9, sp, r9          // T1
// CHECK: add   r9, sp, r9          @ encoding: [0xe9,0x44]
    ADD      r9, r9, sp          // T1
// FIXME: ARMARM says 'add   r9, sp, r9'
// CHECK: add   r9, sp              @ encoding: [0xe9,0x44]
    ADD      sp, sp, r10         // T2
// CHECK: add   sp, r10             @ encoding: [0xd5,0x44]
    ADD      sp, r10, sp         // T2
// CHECK: add   sp, r10             @ encoding: [0xd5,0x44]
    ADD      sp, sp, pc          // T2
// CHECK: add   sp, pc              @ encoding: [0xfd,0x44]

// AND (commutative)
    ANDS     r0, r2, r1          // Must be wide - 3 distinct registers
    ANDS     r2, r2, r1          // Should choose narrow
    ANDS     r2, r1, r2          // Should choose narrow - commutative
    ANDS.W   r0, r0, r1          // Explicitly wide
    ANDS.W   r3, r1, r3  
    AND      r0, r1, r0          // Must use wide encoding as not flag-setting
    ANDS     r7, r7, r1          // Should use narrow
    ANDS     r7, r1, r7          // Commutative
    ANDS     r8, r1, r8          // high registers so must use wide encoding
    ANDS     r8, r8, r1
    ANDS     r0, r8, r0
    ANDS     r1, r1, r8
    ANDS     r2, r2, r1, lsl #1  // Must use wide - shifted register
    ANDS     r0, r1, r0, lsr #1
// CHECK: ands.w  r0, r2, r1              @ encoding: [0x12,0xea,0x01,0x00]
// CHECK: ands    r2, r1                  @ encoding: [0x0a,0x40]
// CHECK: ands    r2, r1                  @ encoding: [0x0a,0x40]
// CHECK: ands.w  r0, r0, r1              @ encoding: [0x10,0xea,0x01,0x00]
// CHECK: ands.w  r3, r1, r3              @ encoding: [0x11,0xea,0x03,0x03]
// CHECK: and.w   r0, r1, r0              @ encoding: [0x01,0xea,0x00,0x00]
// CHECK: ands    r7, r1                  @ encoding: [0x0f,0x40]
// CHECK: ands    r7, r1                  @ encoding: [0x0f,0x40]
// CHECK: ands.w  r8, r1, r8              @ encoding: [0x11,0xea,0x08,0x08]
// CHECK: ands.w  r8, r8, r1              @ encoding: [0x18,0xea,0x01,0x08]
// CHECK: ands.w  r0, r8, r0              @ encoding: [0x18,0xea,0x00,0x00]
// CHECK: ands.w  r1, r1, r8              @ encoding: [0x11,0xea,0x08,0x01]
// CHECK: ands.w  r2, r2, r1, lsl #1      @ encoding: [0x12,0xea,0x41,0x02]
// CHECK: ands.w  r0, r1, r0, lsr #1      @ encoding: [0x11,0xea,0x50,0x00]

    IT EQ
    ANDEQ    r0, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    ANDEQ    r3, r3, r1          // Should choose narrow
    IT EQ
    ANDEQ    r3, r1, r3          // Should choose narrow - commutative
    IT EQ
    ANDEQ.W  r0, r0, r1          // Explicitly wide
    IT EQ
    ANDEQ.W  r2, r1, r2  
    IT EQ
    ANDSEQ   r0, r1, r0          // Must use wide encoding as flag-setting
    IT EQ
    ANDEQ    r7, r7, r1          // Should use narrow
    IT EQ
    ANDEQ    r7, r1, r7          // Commutative
    IT EQ
    ANDEQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    ANDEQ    r8, r8, r1
    IT EQ
    ANDEQ    r4, r8, r4
    IT EQ
    ANDEQ    r4, r4, r8
    IT EQ
    ANDEQ    r0, r0, r1, lsl #1  // Must use wide - shifted register
    IT EQ
    ANDEQ    r5, r1, r5, lsr #1
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r0, r2, r1             @ encoding: [0x02,0xea,0x01,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq    r3, r1                 @ encoding: [0x0b,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq    r3, r1                 @ encoding: [0x0b,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r0, r0, r1             @ encoding: [0x00,0xea,0x01,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r2, r1, r2             @ encoding: [0x01,0xea,0x02,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andseq.w r0, r1, r0             @ encoding: [0x11,0xea,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq    r7, r1                 @ encoding: [0x0f,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq    r7, r1                 @ encoding: [0x0f,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r8, r1, r8             @ encoding: [0x01,0xea,0x08,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r8, r8, r1             @ encoding: [0x08,0xea,0x01,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r4, r8, r4             @ encoding: [0x08,0xea,0x04,0x04]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r4, r4, r8             @ encoding: [0x04,0xea,0x08,0x04]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r0, r0, r1, lsl #1     @ encoding: [0x00,0xea,0x41,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: andeq.w  r5, r1, r5, lsr #1     @ encoding: [0x01,0xea,0x55,0x05]

// EOR (commutative)
    EORS     r0, r2, r1          // Must be wide - 3 distinct registers
    EORS     r5, r5, r1          // Should choose narrow
    EORS     r5, r1, r5          // Should choose narrow - commutative
    EORS.W   r0, r0, r1          // Explicitly wide
    EORS.W   r2, r1, r2  
    EOR      r1, r1, r1          // Must use wide encoding as not flag-setting
    EORS     r7, r7, r1          // Should use narrow
    EORS     r7, r1, r7          // Commutative
    EORS     r8, r1, r8          // high registers so must use wide encoding
    EORS     r8, r8, r1
    EORS     r6, r8, r6
    EORS     r0, r0, r8
    EORS     r2, r2, r1, lsl #1  // Must use wide - shifted register
    EORS     r0, r1, r0, lsr #1
// CHECK: eors.w  r0, r2, r1              @ encoding: [0x92,0xea,0x01,0x00]
// CHECK: eors    r5, r1                  @ encoding: [0x4d,0x40]
// CHECK: eors    r5, r1                  @ encoding: [0x4d,0x40]
// CHECK: eors.w  r0, r0, r1              @ encoding: [0x90,0xea,0x01,0x00]
// CHECK: eors.w  r2, r1, r2              @ encoding: [0x91,0xea,0x02,0x02]
// CHECK: eor.w   r1, r1, r1              @ encoding: [0x81,0xea,0x01,0x01]
// CHECK: eors    r7, r1                  @ encoding: [0x4f,0x40]
// CHECK: eors    r7, r1                  @ encoding: [0x4f,0x40]
// CHECK: eors.w  r8, r1, r8              @ encoding: [0x91,0xea,0x08,0x08]
// CHECK: eors.w  r8, r8, r1              @ encoding: [0x98,0xea,0x01,0x08]
// CHECK: eors.w  r6, r8, r6              @ encoding: [0x98,0xea,0x06,0x06]
// CHECK: eors.w  r0, r0, r8              @ encoding: [0x90,0xea,0x08,0x00]
// CHECK: eors.w  r2, r2, r1, lsl #1      @ encoding: [0x92,0xea,0x41,0x02]
// CHECK: eors.w  r0, r1, r0, lsr #1      @ encoding: [0x91,0xea,0x50,0x00]

    IT EQ
    EOREQ    r3, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    EOREQ    r0, r0, r1          // Should choose narrow
    IT EQ
    EOREQ    r2, r1, r2          // Should choose narrow - commutative
    IT EQ
    EOREQ.W  r3, r3, r1          // Explicitly wide
    IT EQ
    EOREQ.W  r0, r1, r0  
    IT EQ
    EORSEQ   r1, r1, r1          // Must use wide encoding as flag-setting
    IT EQ
    EOREQ    r7, r7, r1          // Should use narrow
    IT EQ
    EOREQ    r7, r1, r7          // Commutative
    IT EQ
    EOREQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    EOREQ    r8, r8, r1
    IT EQ
    EOREQ    r0, r8, r0
    IT EQ
    EOREQ    r3, r3, r8
    IT EQ
    EOREQ    r4, r4, r1, lsl #1  // Must use wide - shifted register
    IT EQ
    EOREQ    r0, r1, r0, lsr #1
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r3, r2, r1             @ encoding: [0x82,0xea,0x01,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq    r0, r1                 @ encoding: [0x48,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq    r2, r1                 @ encoding: [0x4a,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r3, r3, r1             @ encoding: [0x83,0xea,0x01,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r0, r1, r0             @ encoding: [0x81,0xea,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eorseq.w r1, r1, r1             @ encoding: [0x91,0xea,0x01,0x01]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq    r7, r1                 @ encoding: [0x4f,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq    r7, r1                 @ encoding: [0x4f,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r8, r1, r8             @ encoding: [0x81,0xea,0x08,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r8, r8, r1             @ encoding: [0x88,0xea,0x01,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r0, r8, r0             @ encoding: [0x88,0xea,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r3, r3, r8             @ encoding: [0x83,0xea,0x08,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r4, r4, r1, lsl #1     @ encoding: [0x84,0xea,0x41,0x04]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: eoreq.w  r0, r1, r0, lsr #1     @ encoding: [0x81,0xea,0x50,0x00]

// LSL 
    LSLS     r0, r2, r1          // Must be wide - 3 distinct registers
    LSLS     r2, r2, r1          // Should choose narrow
    LSLS     r2, r1, r2          // Should choose wide - not commutative
    LSLS.W   r0, r0, r1          // Explicitly wide
    LSLS.W   r4, r1, r4  
    LSL      r4, r1, r4          // Must use wide encoding as not flag-setting
    LSLS     r7, r7, r1          // Should use narrow
    LSLS     r8, r1, r8          // high registers so must use wide encoding
    LSLS     r8, r8, r1
    LSLS     r3, r8, r3
    LSLS     r5, r5, r8
// CHECK: lsls.w  r0, r2, r1              @ encoding: [0x12,0xfa,0x01,0xf0]
// CHECK: lsls    r2, r1                  @ encoding: [0x8a,0x40]
// CHECK: lsls.w  r2, r1, r2              @ encoding: [0x11,0xfa,0x02,0xf2]
// CHECK: lsls.w  r0, r0, r1              @ encoding: [0x10,0xfa,0x01,0xf0]
// CHECK: lsls.w  r4, r1, r4              @ encoding: [0x11,0xfa,0x04,0xf4]
// CHECK: lsl.w   r4, r1, r4              @ encoding: [0x01,0xfa,0x04,0xf4]
// CHECK: lsls    r7, r1                  @ encoding: [0x8f,0x40]
// CHECK: lsls.w  r8, r1, r8              @ encoding: [0x11,0xfa,0x08,0xf8]
// CHECK: lsls.w  r8, r8, r1              @ encoding: [0x18,0xfa,0x01,0xf8]
// CHECK: lsls.w  r3, r8, r3              @ encoding: [0x18,0xfa,0x03,0xf3]
// CHECK: lsls.w  r5, r5, r8              @ encoding: [0x15,0xfa,0x08,0xf5]

    IT EQ
    LSLEQ    r0, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    LSLEQ    r2, r2, r1          // Should choose narrow
    IT EQ
    LSLEQ    r2, r1, r2          // Should choose wide - not commutative
    IT EQ
    LSLEQ.W  r0, r0, r1          // Explicitly wide
    IT EQ
    LSLEQ.W  r3, r1, r3  
    IT EQ
    LSLSEQ   r4, r1, r4          // Must use wide encoding as flag-setting
    IT EQ
    LSLEQ    r7, r7, r1          // Should use narrow
    IT EQ
    LSLEQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    LSLEQ    r8, r8, r1
    IT EQ
    LSLEQ    r0, r8, r0
    IT EQ
    LSLEQ    r3, r3, r8
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r0, r2, r1             @ encoding: [0x02,0xfa,0x01,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq    r2, r1                 @ encoding: [0x8a,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r2, r1, r2             @ encoding: [0x01,0xfa,0x02,0xf2]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r0, r0, r1             @ encoding: [0x00,0xfa,0x01,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r3, r1, r3             @ encoding: [0x01,0xfa,0x03,0xf3]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lslseq.w r4, r1, r4             @ encoding: [0x11,0xfa,0x04,0xf4]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq    r7, r1                 @ encoding: [0x8f,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r8, r1, r8             @ encoding: [0x01,0xfa,0x08,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r8, r8, r1             @ encoding: [0x08,0xfa,0x01,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r0, r8, r0             @ encoding: [0x08,0xfa,0x00,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsleq.w  r3, r3, r8             @ encoding: [0x03,0xfa,0x08,0xf3]

// LSR 
    LSRS     r6, r2, r1          // Must be wide - 3 distinct registers
    LSRS     r2, r2, r1          // Should choose narrow
    LSRS     r2, r1, r2          // Should choose wide - not commutative
    LSRS.W   r2, r2, r1          // Explicitly wide
    LSRS.W   r3, r1, r3  
    LSR      r4, r1, r4          // Must use wide encoding as not flag-setting
    LSRS     r7, r7, r1          // Should use narrow
    LSRS     r8, r1, r8          // high registers so must use wide encoding
    LSRS     r8, r8, r1
    LSRS     r2, r8, r2
    LSRS     r5, r5, r8
// CHECK: lsrs.w  r6, r2, r1              @ encoding: [0x32,0xfa,0x01,0xf6]
// CHECK: lsrs    r2, r1                  @ encoding: [0xca,0x40]
// CHECK: lsrs.w  r2, r1, r2              @ encoding: [0x31,0xfa,0x02,0xf2]
// CHECK: lsrs.w  r2, r2, r1              @ encoding: [0x32,0xfa,0x01,0xf2]
// CHECK: lsrs.w  r3, r1, r3              @ encoding: [0x31,0xfa,0x03,0xf3]
// CHECK: lsr.w   r4, r1, r4              @ encoding: [0x21,0xfa,0x04,0xf4]
// CHECK: lsrs    r7, r1                  @ encoding: [0xcf,0x40]
// CHECK: lsrs.w  r8, r1, r8              @ encoding: [0x31,0xfa,0x08,0xf8]
// CHECK: lsrs.w  r8, r8, r1              @ encoding: [0x38,0xfa,0x01,0xf8]
// CHECK: lsrs.w  r2, r8, r2              @ encoding: [0x38,0xfa,0x02,0xf2]
// CHECK: lsrs.w  r5, r5, r8              @ encoding: [0x35,0xfa,0x08,0xf5]

    IT EQ
    LSREQ    r6, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    LSREQ    r7, r7, r1          // Should choose narrow
    IT EQ
    LSREQ    r7, r1, r7          // Should choose wide - not commutative
    IT EQ
    LSREQ.W  r7, r7, r1          // Explicitly wide
    IT EQ
    LSREQ.W  r2, r1, r2  
    IT EQ
    LSRSEQ   r0, r1, r0          // Must use wide encoding as flag-setting
    IT EQ
    LSREQ    r7, r7, r1          // Should use narrow
    IT EQ
    LSREQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    LSREQ    r8, r8, r1
    IT EQ
    LSREQ    r1, r8, r1
    IT EQ
    LSREQ    r4, r4, r8
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r6, r2, r1             @ encoding: [0x22,0xfa,0x01,0xf6]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq    r7, r1                 @ encoding: [0xcf,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r7, r1, r7             @ encoding: [0x21,0xfa,0x07,0xf7]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r7, r7, r1             @ encoding: [0x27,0xfa,0x01,0xf7]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r2, r1, r2             @ encoding: [0x21,0xfa,0x02,0xf2]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsrseq.w r0, r1, r0             @ encoding: [0x31,0xfa,0x00,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq    r7, r1                 @ encoding: [0xcf,0x40]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r8, r1, r8             @ encoding: [0x21,0xfa,0x08,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r8, r8, r1             @ encoding: [0x28,0xfa,0x01,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r1, r8, r1             @ encoding: [0x28,0xfa,0x01,0xf1]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: lsreq.w  r4, r4, r8             @ encoding: [0x24,0xfa,0x08,0xf4]

// ASR 
    ASRS     r7, r6, r5          // Must be wide - 3 distinct registers
    ASRS     r0, r0, r1          // Should choose narrow
    ASRS     r0, r1, r0          // Should choose wide - not commutative
    ASRS.W   r3, r3, r1          // Explicitly wide
    ASRS.W   r1, r1, r1  
    ASR      r0, r1, r0          // Must use wide encoding as not flag-setting
    ASRS     r7, r7, r1          // Should use narrow
    ASRS     r8, r1, r8          // high registers so must use wide encoding
    ASRS     r8, r8, r1
    ASRS     r5, r8, r5
    ASRS     r5, r5, r8
// CHECK: asrs.w  r7, r6, r5              @ encoding: [0x56,0xfa,0x05,0xf7]
// CHECK: asrs    r0, r1                  @ encoding: [0x08,0x41]
// CHECK: asrs.w  r0, r1, r0              @ encoding: [0x51,0xfa,0x00,0xf0]
// CHECK: asrs.w  r3, r3, r1              @ encoding: [0x53,0xfa,0x01,0xf3]
// CHECK: asrs.w  r1, r1, r1              @ encoding: [0x51,0xfa,0x01,0xf1]
// CHECK: asr.w   r0, r1, r0              @ encoding: [0x41,0xfa,0x00,0xf0]
// CHECK: asrs    r7, r1                  @ encoding: [0x0f,0x41]
// CHECK: asrs.w  r8, r1, r8              @ encoding: [0x51,0xfa,0x08,0xf8]
// CHECK: asrs.w  r8, r8, r1              @ encoding: [0x58,0xfa,0x01,0xf8]
// CHECK: asrs.w  r5, r8, r5              @ encoding: [0x58,0xfa,0x05,0xf5]
// CHECK: asrs.w  r5, r5, r8              @ encoding: [0x55,0xfa,0x08,0xf5]

    IT EQ
    ASREQ    r0, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    ASREQ    r2, r2, r1          // Should choose narrow
    IT EQ
    ASREQ    r1, r2, r1          // Should choose wide - not commutative
    IT EQ
    ASREQ.W  r4, r4, r1          // Explicitly wide
    IT EQ
    ASREQ.W  r6, r1, r6  
    IT EQ
    ASRSEQ   r3, r1, r3          // Must use wide encoding as flag-setting
    IT EQ
    ASREQ    r7, r7, r1          // Should use narrow
    IT EQ
    ASREQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    ASREQ    r8, r8, r1
    IT EQ
    ASREQ    r1, r8, r1
    IT EQ
    ASREQ    r3, r3, r8
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r0, r2, r1             @ encoding: [0x42,0xfa,0x01,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq    r2, r1                 @ encoding: [0x0a,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r1, r2, r1             @ encoding: [0x42,0xfa,0x01,0xf1]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r4, r4, r1             @ encoding: [0x44,0xfa,0x01,0xf4]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r6, r1, r6             @ encoding: [0x41,0xfa,0x06,0xf6]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asrseq.w r3, r1, r3             @ encoding: [0x51,0xfa,0x03,0xf3]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq    r7, r1                 @ encoding: [0x0f,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r8, r1, r8             @ encoding: [0x41,0xfa,0x08,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r8, r8, r1             @ encoding: [0x48,0xfa,0x01,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r1, r8, r1             @ encoding: [0x48,0xfa,0x01,0xf1]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: asreq.w  r3, r3, r8             @ encoding: [0x43,0xfa,0x08,0xf3]

// ADC (commutative)
    ADCS     r5, r2, r1          // Must be wide - 3 distinct registers
    ADCS     r5, r5, r1          // Should choose narrow  
    ADCS     r3, r1, r3          // Should choose narrow - commutative
    ADCS.W   r2, r2, r1          // Explicitly wide
    ADCS.W   r3, r1, r3  
    ADC      r0, r1, r0          // Must use wide encoding as not flag-setting
    ADCS     r7, r7, r1          // Should use narrow
    ADCS     r7, r1, r7          // Commutative
    ADCS     r8, r1, r8          // high registers so must use wide encoding
    ADCS     r8, r8, r1
    ADCS     r5, r8, r5
    ADCS     r2, r2, r8
    ADCS     r3, r3, r1, lsl #1  // Must use wide - shifted register
    ADCS     r4, r1, r4, lsr #1
// CHECK: adcs.w  r5, r2, r1              @ encoding: [0x52,0xeb,0x01,0x05]
// CHECK: adcs    r5, r1                  @ encoding: [0x4d,0x41]
// CHECK: adcs    r3, r1                  @ encoding: [0x4b,0x41]
// CHECK: adcs.w  r2, r2, r1              @ encoding: [0x52,0xeb,0x01,0x02]
// CHECK: adcs.w  r3, r1, r3              @ encoding: [0x51,0xeb,0x03,0x03]
// CHECK: adc.w   r0, r1, r0              @ encoding: [0x41,0xeb,0x00,0x00]
// CHECK: adcs    r7, r1                  @ encoding: [0x4f,0x41]
// CHECK: adcs    r7, r1                  @ encoding: [0x4f,0x41]
// CHECK: adcs.w  r8, r1, r8              @ encoding: [0x51,0xeb,0x08,0x08]
// CHECK: adcs.w  r8, r8, r1              @ encoding: [0x58,0xeb,0x01,0x08]
// CHECK: adcs.w  r5, r8, r5              @ encoding: [0x58,0xeb,0x05,0x05]
// CHECK: adcs.w  r2, r2, r8              @ encoding: [0x52,0xeb,0x08,0x02]
// CHECK: adcs.w  r3, r3, r1, lsl #1      @ encoding: [0x53,0xeb,0x41,0x03]
// CHECK: adcs.w  r4, r1, r4, lsr #1      @ encoding: [0x51,0xeb,0x54,0x04]

    IT EQ
    ADCEQ    r1, r2, r3          // Must be wide - 3 distinct registers
    IT EQ
    ADCEQ    r1, r1, r1          // Should choose narrow
    IT EQ
    ADCEQ    r3, r1, r3          // Should choose narrow - commutative
    IT EQ
    ADCEQ.W  r3, r3, r1          // Explicitly wide
    IT EQ
    ADCEQ.W  r0, r1, r0  
    IT EQ
    ADCSEQ   r3, r1, r3          // Must use wide encoding as flag-setting
    IT EQ
    ADCEQ    r7, r7, r1          // Should use narrow 
    IT EQ
    ADCEQ    r7, r1, r7          // Commutative
    IT EQ
    ADCEQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    ADCEQ    r8, r8, r1
    IT EQ
    ADCEQ    r3, r8, r3
    IT EQ
    ADCEQ    r1, r1, r8
    IT EQ
    ADCEQ    r2, r2, r1, lsl #1  // Must use wide - shifted register
    IT EQ
    ADCEQ    r1, r1, r1, lsr #1
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r1, r2, r3             @ encoding: [0x42,0xeb,0x03,0x01]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq    r1, r1                 @ encoding: [0x49,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq    r3, r1                 @ encoding: [0x4b,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r3, r3, r1             @ encoding: [0x43,0xeb,0x01,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r0, r1, r0             @ encoding: [0x41,0xeb,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adcseq.w r3, r1, r3             @ encoding: [0x51,0xeb,0x03,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq    r7, r1                 @ encoding: [0x4f,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq    r7, r1                 @ encoding: [0x4f,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r8, r1, r8             @ encoding: [0x41,0xeb,0x08,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r8, r8, r1             @ encoding: [0x48,0xeb,0x01,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r3, r8, r3             @ encoding: [0x48,0xeb,0x03,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r1, r1, r8             @ encoding: [0x41,0xeb,0x08,0x01]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r2, r2, r1, lsl #1     @ encoding: [0x42,0xeb,0x41,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: adceq.w  r1, r1, r1, lsr #1     @ encoding: [0x41,0xeb,0x51,0x01]

// SBC 
    SBCS     r3, r2, r1          // Must be wide - 3 distinct registers
    SBCS     r4, r4, r1          // Should choose narrow  
    SBCS     r1, r4, r1          // Should choose wide - not commutative  
    SBCS.W   r4, r4, r1          // Explicitly wide
    SBCS.W   r2, r1, r2  
    SBC      r0, r1, r0          // Must use wide encoding as not flag-setting
    SBCS     r7, r7, r1          // Should use narrow
    SBCS     r8, r1, r8          // high registers so must use wide encoding
    SBCS     r8, r8, r1
    SBCS     r4, r8, r4
    SBCS     r3, r3, r8
    SBCS     r2, r2, r1, lsl #1  // Must use wide - shifted register
    SBCS     r5, r1, r5, lsr #1
// CHECK: sbcs.w  r3, r2, r1              @ encoding: [0x72,0xeb,0x01,0x03]
// CHECK: sbcs    r4, r1                  @ encoding: [0x8c,0x41]
// CHECK: sbcs.w  r1, r4, r1              @ encoding: [0x74,0xeb,0x01,0x01]
// CHECK: sbcs.w  r4, r4, r1              @ encoding: [0x74,0xeb,0x01,0x04]
// CHECK: sbcs.w  r2, r1, r2              @ encoding: [0x71,0xeb,0x02,0x02]
// CHECK: sbc.w   r0, r1, r0              @ encoding: [0x61,0xeb,0x00,0x00]
// CHECK: sbcs    r7, r1                  @ encoding: [0x8f,0x41]
// CHECK: sbcs.w  r8, r1, r8              @ encoding: [0x71,0xeb,0x08,0x08]
// CHECK: sbcs.w  r8, r8, r1              @ encoding: [0x78,0xeb,0x01,0x08]
// CHECK: sbcs.w  r4, r8, r4              @ encoding: [0x78,0xeb,0x04,0x04]
// CHECK: sbcs.w  r3, r3, r8              @ encoding: [0x73,0xeb,0x08,0x03]
// CHECK: sbcs.w  r2, r2, r1, lsl #1      @ encoding: [0x72,0xeb,0x41,0x02]
// CHECK: sbcs.w  r5, r1, r5, lsr #1      @ encoding: [0x71,0xeb,0x55,0x05]

    IT EQ
    SBCEQ    r5, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    SBCEQ    r5, r5, r1          // Should choose narrow
    IT EQ
    SBCEQ    r1, r5, r1          // Should choose narrow
    IT EQ
    SBCEQ.W  r5, r5, r1          // Explicitly wide
    IT EQ
    SBCEQ.W  r0, r1, r0  
    IT EQ
    SBCSEQ   r2, r1, r2          // Must use wide encoding as flag-setting
    IT EQ
    SBCEQ    r7, r7, r1          // Should use narrow 
    IT EQ
    SBCEQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    SBCEQ    r8, r8, r1
    IT EQ
    SBCEQ    r7, r8, r7
    IT EQ
    SBCEQ    r7, r7, r8
    IT EQ
    SBCEQ    r2, r2, r1, lsl #1  // Must use wide - shifted register
    IT EQ
    SBCEQ    r5, r1, r5, lsr #1
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r5, r2, r1             @ encoding: [0x62,0xeb,0x01,0x05]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq    r5, r1                 @ encoding: [0x8d,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r1, r5, r1             @ encoding: [0x65,0xeb,0x01,0x01]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r5, r5, r1             @ encoding: [0x65,0xeb,0x01,0x05]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r0, r1, r0             @ encoding: [0x61,0xeb,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbcseq.w r2, r1, r2             @ encoding: [0x71,0xeb,0x02,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq    r7, r1                 @ encoding: [0x8f,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r8, r1, r8             @ encoding: [0x61,0xeb,0x08,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r8, r8, r1             @ encoding: [0x68,0xeb,0x01,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r7, r8, r7             @ encoding: [0x68,0xeb,0x07,0x07]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r7, r7, r8             @ encoding: [0x67,0xeb,0x08,0x07]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r2, r2, r1, lsl #1     @ encoding: [0x62,0xeb,0x41,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: sbceq.w  r5, r1, r5, lsr #1     @ encoding: [0x61,0xeb,0x55,0x05]

// ROR 
    RORS     r3, r2, r1          // Must be wide - 3 distinct registers
    RORS     r0, r0, r1          // Should choose narrow
    RORS     r1, r0, r1          // Should choose wide - not commutative
    RORS.W   r2, r2, r1          // Explicitly wide
    RORS.W   r2, r1, r2  
    ROR      r5, r1, r5          // Must use wide encoding as not flag-setting
    RORS     r7, r7, r1          // Should use narrow
    RORS     r8, r1, r8          // high registers so must use wide encoding
    RORS     r8, r8, r1
    RORS     r6, r8, r6
    RORS     r6, r6, r8
// CHECK: rors.w  r3, r2, r1              @ encoding: [0x72,0xfa,0x01,0xf3]
// CHECK: rors    r0, r1                  @ encoding: [0xc8,0x41]
// CHECK: rors.w  r1, r0, r1              @ encoding: [0x70,0xfa,0x01,0xf1]
// CHECK: rors.w  r2, r2, r1              @ encoding: [0x72,0xfa,0x01,0xf2]
// CHECK: rors.w  r2, r1, r2              @ encoding: [0x71,0xfa,0x02,0xf2]
// CHECK: ror.w   r5, r1, r5              @ encoding: [0x61,0xfa,0x05,0xf5]
// CHECK: rors    r7, r1                  @ encoding: [0xcf,0x41]
// CHECK: rors.w  r8, r1, r8              @ encoding: [0x71,0xfa,0x08,0xf8]
// CHECK: rors.w  r8, r8, r1              @ encoding: [0x78,0xfa,0x01,0xf8]
// CHECK: rors.w  r6, r8, r6              @ encoding: [0x78,0xfa,0x06,0xf6]
// CHECK: rors.w  r6, r6, r8              @ encoding: [0x76,0xfa,0x08,0xf6]

    IT EQ
    ROREQ    r4, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    ROREQ    r4, r4, r1          // Should choose narrow
    IT EQ
    ROREQ    r1, r4, r1          // Should choose wide - not commutative
    IT EQ
    ROREQ.W  r4, r4, r1          // Explicitly wide
    IT EQ
    ROREQ.W  r0, r1, r0  
    IT EQ
    RORSEQ   r0, r1, r0          // Must use wide encoding as flag-setting
    IT EQ
    ROREQ    r7, r7, r1          // Should use narrow
    IT EQ
    ROREQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    ROREQ    r8, r8, r1
    IT EQ
    ROREQ    r3, r8, r3
    IT EQ
    ROREQ    r1, r1, r8
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r4, r2, r1             @ encoding: [0x62,0xfa,0x01,0xf4]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq    r4, r1                 @ encoding: [0xcc,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r1, r4, r1             @ encoding: [0x64,0xfa,0x01,0xf1]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r4, r4, r1             @ encoding: [0x64,0xfa,0x01,0xf4]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r0, r1, r0             @ encoding: [0x61,0xfa,0x00,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: rorseq.w r0, r1, r0             @ encoding: [0x71,0xfa,0x00,0xf0]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq    r7, r1                 @ encoding: [0xcf,0x41]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r8, r1, r8             @ encoding: [0x61,0xfa,0x08,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r8, r8, r1             @ encoding: [0x68,0xfa,0x01,0xf8]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r3, r8, r3             @ encoding: [0x68,0xfa,0x03,0xf3]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: roreq.w  r1, r1, r8             @ encoding: [0x61,0xfa,0x08,0xf1]

// TST - only two register version available
// RSB - only two register version available
// CMP - only two register version available
// CMN - only two register version available

// ORR (commutative)
    ORRS     r7, r2, r1          // Must be wide - 3 distinct registers
    ORRS     r2, r2, r1          // Should choose narrow
    ORRS     r3, r1, r3          // Should choose narrow - commutative
    ORRS.W   r4, r4, r1          // Explicitly wide
    ORRS.W   r5, r1, r5  
    ORR      r2, r1, r2          // Must use wide encoding as not flag-setting
    ORRS     r7, r7, r1          // Should use narrow
    ORRS     r7, r1, r7          // Commutative
    ORRS     r8, r1, r8          // high registers so must use wide encoding
    ORRS     r8, r8, r1
    ORRS     r1, r8, r1
    ORRS     r0, r0, r8
    ORRS     r1, r1, r1, lsl #1  // Must use wide - shifted register
    ORRS     r0, r1, r0, lsr #1
// CHECK: orrs.w  r7, r2, r1              @ encoding: [0x52,0xea,0x01,0x07]
// CHECK: orrs    r2, r1                  @ encoding: [0x0a,0x43]
// CHECK: orrs    r3, r1                  @ encoding: [0x0b,0x43]
// CHECK: orrs.w  r4, r4, r1              @ encoding: [0x54,0xea,0x01,0x04]
// CHECK: orrs.w  r5, r1, r5              @ encoding: [0x51,0xea,0x05,0x05]
// CHECK: orr.w   r2, r1, r2              @ encoding: [0x41,0xea,0x02,0x02]
// CHECK: orrs    r7, r1                  @ encoding: [0x0f,0x43]
// CHECK: orrs    r7, r1                  @ encoding: [0x0f,0x43]
// CHECK: orrs.w  r8, r1, r8              @ encoding: [0x51,0xea,0x08,0x08]
// CHECK: orrs.w  r8, r8, r1              @ encoding: [0x58,0xea,0x01,0x08]
// CHECK: orrs.w  r1, r8, r1              @ encoding: [0x58,0xea,0x01,0x01]
// CHECK: orrs.w  r0, r0, r8              @ encoding: [0x50,0xea,0x08,0x00]
// CHECK: orrs.w  r1, r1, r1, lsl #1      @ encoding: [0x51,0xea,0x41,0x01]
// CHECK: orrs.w  r0, r1, r0, lsr #1      @ encoding: [0x51,0xea,0x50,0x00]

    IT EQ
    ORREQ    r0, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    ORREQ    r5, r5, r1          // Should choose narrow
    IT EQ
    ORREQ    r5, r1, r5          // Should choose narrow - commutative
    IT EQ
    ORREQ.W  r2, r2, r1          // Explicitly wide
    IT EQ
    ORREQ.W  r3, r1, r3  
    IT EQ
    ORRSEQ   r4, r1, r4          // Must use wide encoding as flag-setting
    IT EQ
    ORREQ    r7, r7, r1          // Should use narrow
    IT EQ
    ORREQ    r7, r1, r7          // Commutative
    IT EQ
    ORREQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    ORREQ    r8, r8, r1
    IT EQ
    ORREQ    r0, r8, r0
    IT EQ
    ORREQ    r0, r0, r8
    IT EQ
    ORREQ    r2, r2, r1, lsl #1  // Must use wide - shifted register
    IT EQ
    ORREQ    r2, r1, r2, lsr #1
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r0, r2, r1             @ encoding: [0x42,0xea,0x01,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq    r5, r1                 @ encoding: [0x0d,0x43]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq    r5, r1                 @ encoding: [0x0d,0x43]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r2, r2, r1             @ encoding: [0x42,0xea,0x01,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r3, r1, r3             @ encoding: [0x41,0xea,0x03,0x03]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orrseq.w r4, r1, r4             @ encoding: [0x51,0xea,0x04,0x04]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq    r7, r1                 @ encoding: [0x0f,0x43]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq    r7, r1                 @ encoding: [0x0f,0x43]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r8, r1, r8             @ encoding: [0x41,0xea,0x08,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r8, r8, r1             @ encoding: [0x48,0xea,0x01,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r0, r8, r0             @ encoding: [0x48,0xea,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r0, r0, r8             @ encoding: [0x40,0xea,0x08,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r2, r2, r1, lsl #1     @ encoding: [0x42,0xea,0x41,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: orreq.w  r2, r1, r2, lsr #1     @ encoding: [0x41,0xea,0x52,0x02]

// MUL - not affected by this change

// BIC 
    BICS     r3, r2, r1          // Must be wide - 3 distinct registers
    BICS     r2, r2, r1          // Should choose narrow  
    BICS     r1, r2, r1          // Should choose wide - not commutative  
    BICS.W   r2, r2, r1          // Explicitly wide
    BICS.W   r0, r1, r0  
    BIC      r0, r1, r0          // Must use wide encoding as not flag-setting
    BICS     r7, r7, r1          // Should use narrow
    BICS     r8, r1, r8          // high registers so must use wide encoding
    BICS     r8, r8, r1
    BICS     r7, r8, r7
    BICS     r5, r5, r8
    BICS     r3, r3, r1, lsl #1  // Must use wide - shifted register
    BICS     r4, r1, r4, lsr #1
// CHECK: bics.w  r3, r2, r1              @ encoding: [0x32,0xea,0x01,0x03]
// CHECK: bics    r2, r1                  @ encoding: [0x8a,0x43]
// CHECK: bics.w  r1, r2, r1              @ encoding: [0x32,0xea,0x01,0x01]
// CHECK: bics.w  r2, r2, r1              @ encoding: [0x32,0xea,0x01,0x02]
// CHECK: bics.w  r0, r1, r0              @ encoding: [0x31,0xea,0x00,0x00]
// CHECK: bic.w   r0, r1, r0              @ encoding: [0x21,0xea,0x00,0x00]
// CHECK: bics    r7, r1                  @ encoding: [0x8f,0x43]
// CHECK: bics.w  r8, r1, r8              @ encoding: [0x31,0xea,0x08,0x08]
// CHECK: bics.w  r8, r8, r1              @ encoding: [0x38,0xea,0x01,0x08]
// CHECK: bics.w  r7, r8, r7              @ encoding: [0x38,0xea,0x07,0x07]
// CHECK: bics.w  r5, r5, r8              @ encoding: [0x35,0xea,0x08,0x05]
// CHECK: bics.w  r3, r3, r1, lsl #1      @ encoding: [0x33,0xea,0x41,0x03]
// CHECK: bics.w  r4, r1, r4, lsr #1      @ encoding: [0x31,0xea,0x54,0x04]

    IT EQ
    BICEQ    r0, r2, r1          // Must be wide - 3 distinct registers
    IT EQ
    BICEQ    r5, r5, r1          // Should choose narrow
    IT EQ
    BICEQ    r1, r5, r1          // Should choose wide - not commutative
    IT EQ
    BICEQ.W  r4, r4, r1          // Explicitly wide
    IT EQ
    BICEQ.W  r2, r1, r2  
    IT EQ
    BICSEQ   r5, r1, r5          // Must use wide encoding as flag-setting
    IT EQ
    BICEQ    r7, r7, r1          // Should use narrow 
    IT EQ
    BICEQ    r8, r1, r8          // high registers so must use wide encoding
    IT EQ
    BICEQ    r8, r8, r1
    IT EQ
    BICEQ    r0, r8, r0
    IT EQ
    BICEQ    r2, r2, r8
    IT EQ
    BICEQ    r4, r4, r1, lsl #1  // Must use wide - shifted register
    IT EQ
    BICEQ    r5, r1, r5, lsr #1
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r0, r2, r1             @ encoding: [0x22,0xea,0x01,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq    r5, r1                 @ encoding: [0x8d,0x43]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r1, r5, r1             @ encoding: [0x25,0xea,0x01,0x01]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r4, r4, r1             @ encoding: [0x24,0xea,0x01,0x04]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r2, r1, r2             @ encoding: [0x21,0xea,0x02,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: bicseq.w r5, r1, r5             @ encoding: [0x31,0xea,0x05,0x05]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq    r7, r1                 @ encoding: [0x8f,0x43]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r8, r1, r8             @ encoding: [0x21,0xea,0x08,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r8, r8, r1             @ encoding: [0x28,0xea,0x01,0x08]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r0, r8, r0             @ encoding: [0x28,0xea,0x00,0x00]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r2, r2, r8             @ encoding: [0x22,0xea,0x08,0x02]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r4, r4, r1, lsl #1     @ encoding: [0x24,0xea,0x41,0x04]
// CHECK: it eq                           @ encoding: [0x08,0xbf]
// CHECK: biceq.w  r5, r1, r5, lsr #1     @ encoding: [0x21,0xea,0x55,0x05]

// CMN - only two register version available
