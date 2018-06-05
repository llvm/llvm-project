@ RUN: not llvm-mc %s -triple thumbv7-linux-gnueabi -filetype=obj -o /dev/null 2>&1 | FileCheck %s

// Thumb2 unconditional branch has a range of +- 16 Megabytes. The
// conditional branch has a range of +- 1 Megabyte. We should give
// an error message if we evaluate the expression at assembly
// time and it is out of range.

        .syntax unified
        .thumb
        b.w end
        .space 0xfffffe
end:
        b.w end2
        .space 0xfffffe
        .global end2
end2:

// branch to arm function uses relocation
        b.w end3
        .space 0x1000000
        .global end3
        .type end3, %function
        .arm
end3:   bx lr
        .thumb

// branch to thumb function is resolved at assembly time
// CHECK-NOT: error
// CHECK: [[@LINE+2]]:{{[0-9]}}: error: Relocation out of range
// CHECK-LABEL: b.w end4
        b.w end4
        .space 0x1000000
        .thumb_func
end4:

        beq.w end5
        .space 0xffffc
end5:

// conditional branch to arm function uses relocation
        beq.w end6
        .arm
        .type end6, %function
        .space 0x100000
end6:   bx lr
        .thumb

// conditional branch to thumb function resolved at assembly time
// CHECK-NOT: error
// CHECK: [[@LINE+2]]:{{[0-9]}}: error: Relocation out of range
// CHECK-LABEL: beq.w end7
        beq.w end7
        .space 0x100000
end7:

start:
        .space 0xfffffc
        b.w start

        .arm
        .global start2
        .type start2, %function
start2:
        .space 0x1000000
        .thumb
// branch to arm function uses relocation
        b.w start2

start3:
        .space 0x1000000
// branch to thumb function resolved at assembly time
// CHECK-NOT: error
// CHECK: [[@LINE+2]]:{{[0-9]}}: error: Relocation out of range
// CHECK-LABEL: b.w start3
        b.w start3

start4:
        .space 0xffffc
        b.w start4

        .arm
        .global start5
        .type start5, %function
start5:
        .space 0x100000
        .thumb
// conditional branch to arm function uses relocation
        beq.w start5

start6:
        .space 0x100000
// branch to thumb function resolved at assembly time
// CHECK-NOT: error
// CHECK: [[@LINE+2]]:{{[0-9]}}: error: Relocation out of range
// CHECK-LABEL: beq.w start6
        beq.w start6
