// REQUIRES: asserts
// Test that sections with many .prefalign fragments converge in a small
// number of relaxation steps (not O(N) steps). Without the layoutSection
// fix, each relaxOnce inner iteration would only correctly resolve one
// PrefAlign fragment (because subsequent fragments see stale offsets),
// leading to O(N) iterations. With the fix, layoutSection recomputes all
// PrefAlign fragments using the tracked offset, converging in 1 iteration.

// RUN: llvm-mc -filetype=obj -triple x86_64 --stats %s -o %t 2>&1 \
// RUN:   | FileCheck %s
// CHECK: 1 assembler - Number of assembler layout and relaxation steps

// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DIS %s

.section .text,"ax",@progbits
.byte 0

// DIS:       8: nop
.prefalign 16, .Lend0, nop
.rept 5
nop
.endr
.Lend0:

// DIS:      10: nop
.prefalign 16, .Lend1, nop
.rept 5
nop
.endr
.Lend1:

// DIS:      18: nop
.prefalign 16, .Lend2, nop
.rept 5
nop
.endr
.Lend2:

// DIS:      20: nop
.prefalign 16, .Lend3, nop
.rept 5
nop
.endr
.Lend3:

// DIS:      28: nop
.prefalign 16, .Lend4, nop
.rept 5
nop
.endr
.Lend4:

// DIS:      30: nop
.prefalign 16, .Lend5, nop
.rept 5
nop
.endr
.Lend5:

// DIS:      38: nop
.prefalign 16, .Lend6, nop
.rept 5
nop
.endr
.Lend6:

// DIS:      40: nop
.prefalign 16, .Lend7, nop
.rept 5
nop
.endr
.Lend7:

// DIS:      48: nop
.prefalign 16, .Lend8, nop
.rept 5
nop
.endr
.Lend8:

// DIS:      50: nop
.prefalign 16, .Lend9, nop
.rept 5
nop
.endr
.Lend9:
