// RUN: llvm-mc -triple thumbv6m -show-encoding < %s | FileCheck %s

  .text
  .thumb

  // Check that the correct encoding of the add and sub instructions is
  // selected, for all combinations of flag-setting, condition and 2- or
  // 3-operand syntax.

  .arch armv6-m
  add r0, r0, r1     // T2
  add r0, r1         // T2
  adds r0, r0, r1    // T1
  adds r0, r1        // T1
// CHECK: add     r0, r1                  @ encoding: [0x08,0x44]
// CHECK: add     r0, r1                  @ encoding: [0x08,0x44]
// CHECK: adds    r0, r0, r1              @ encoding: [0x40,0x18]
// CHECK: adds    r0, r0, r1              @ encoding: [0x40,0x18]

  .arch armv7-m
  add r0, r0, r1     // T2, T3
  add r0, r1         // T2, T3
  adds r0, r0, r1    // T1, T3
  adds r0, r1        // T1, T3
// CHECK: add     r0, r1                  @ encoding: [0x08,0x44]
// CHECK: add     r0, r1                  @ encoding: [0x08,0x44]
// CHECK: adds    r0, r0, r1              @ encoding: [0x40,0x18]
// CHECK: adds    r0, r0, r1              @ encoding: [0x40,0x18]

  itttt eq
// CHECK: itttt   eq                      @ encoding: [0x01,0xbf]
  addeq r0, r0, r1   // T1, T2, T3
  addeq r0, r1       // T2, T1, T3
  addseq r0, r0, r1  // T3
  addseq r0, r1      // T3
  // NOTE: Both T1 and T2 are valid for these two instructions, which one is
  // the preferred varies depending on whether the 2- or 3-operand syntax was
  // used.
// CHECK: addeq   r0, r0, r1              @ encoding: [0x40,0x18]
// CHECK: addeq   r0, r1                  @ encoding: [0x08,0x44]
// CHECK: addseq.w        r0, r0, r1      @ encoding: [0x10,0xeb,0x01,0x00]
// CHECK: addseq.w        r0, r0, r1      @ encoding: [0x10,0xeb,0x01,0x00]

  .arch armv6-m
  // NOTE: There is no non-flag-setting sub instruction for v6-M
  subs r0, r0, r1    // T1, T2
  subs r0, r1        // T1, T2
// CHECK: subs    r0, r0, r1              @ encoding: [0x40,0x1a]
// CHECK: subs    r0, r0, r1              @ encoding: [0x40,0x1a]

  .arch armv7-m
  sub r0, r0, r1     // T2
  sub r0, r1         // T2
  subs r0, r0, r1    // T1, T2
  subs r0, r1        // T1, T2
// CHECK: sub.w   r0, r0, r1              @ encoding: [0xa0,0xeb,0x01,0x00]
// CHECK: sub.w   r0, r0, r1              @ encoding: [0xa0,0xeb,0x01,0x00]
// CHECK: subs    r0, r0, r1              @ encoding: [0x40,0x1a]
// CHECK: subs    r0, r0, r1              @ encoding: [0x40,0x1a]

  itttt eq
// CHECK: itttt   eq                      @ encoding: [0x01,0xbf]
  subeq r0, r0, r1   // T1, T2
  subeq r0, r1       // T1, T2
  subseq r0, r0, r1  // T2
  subseq r0, r1      // T2
// CHECK: subeq   r0, r0, r1              @ encoding: [0x40,0x1a]
// CHECK: subeq   r0, r0, r1              @ encoding: [0x40,0x1a]
// CHECK: subseq.w        r0, r0, r1      @ encoding: [0xb0,0xeb,0x01,0x00]
// CHECK: subseq.w        r0, r0, r1      @ encoding: [0xb0,0xeb,0x01,0x00]
