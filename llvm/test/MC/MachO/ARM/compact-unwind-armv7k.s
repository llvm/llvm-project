@ RUN: llvm-mc -triple=thumbv7k-apple-watchos2.0.0 -filetype=obj -o %t < %s && llvm-objdump --unwind-info %t | FileCheck %s

@ CHECK: Contents of __compact_unwind section:

        .syntax unified
        .align        2
        .code        16

@ CHECK-LABEL: start: {{.*}} _test_r4_r5_r6
@ CHECK: compact encoding: 0x01000007
        .thumb_func        _test_r4_r5_r6
_test_r4_r5_r6:
        .cfi_startproc
        push        {r4, r5, r6, r7, lr}
        add        r7, sp, #12
        sub        sp, #16
        .cfi_def_cfa r7, 8
        .cfi_offset lr, -4
        .cfi_offset r7, -8
        .cfi_offset r6, -12
        .cfi_offset r5, -16
        .cfi_offset r4, -20
        .cfi_endproc


@ CHECK-LABEL: start: {{.*}} _test_r4_r5_r10_r11
@ CHECK: compact encoding: 0x01000063
        .thumb_func        _test_r4_r5_r10_r11
_test_r4_r5_r10_r11:
        .cfi_startproc
        push        {r4, r5, r7, lr}
        add        r7, sp, #8
        .cfi_def_cfa r7, 8
        .cfi_offset lr, -4
        .cfi_offset r7, -8
        .cfi_offset r5, -12
        .cfi_offset r4, -16
        push.w        {r10, r11}
        .cfi_offset r11, -20
        .cfi_offset r10, -24
        .cfi_endproc


@ CHECK-LABEL: start: {{.*}} _test_d8
@ CHECK: compact encoding: 0x02000000
        .thumb_func        _test_d8
_test_d8:
        .cfi_startproc
        push        {r7, lr}
        mov        r7, sp
        .cfi_def_cfa r7, 8
        .cfi_offset lr, -4
        .cfi_offset r7, -8
        vpush        {d8}
        .cfi_offset d8, -16
        .cfi_endproc


@ CHECK-LABEL: start: {{.*}} _test_d8_d10_d12_d14
@ CHECK: compact encoding: 0x02000300
        .thumb_func        _test_d8_d10_d12_d14
_test_d8_d10_d12_d14:
        .cfi_startproc
        push        {r7, lr}
        mov        r7, sp
        .cfi_def_cfa r7, 8
        .cfi_offset lr, -4
        .cfi_offset r7, -8
        vpush        {d14}
        vpush        {d12}
        vpush        {d10}
        vpush        {d8}
        .cfi_offset d14, -16
        .cfi_offset d12, -24
        .cfi_offset d10, -32
        .cfi_offset d8, -40
        .cfi_endproc

@ CHECK-LABEL: start: {{.*}} _test_varargs
@ CHECK: compact encoding: 0x01c00001
        .thumb_func        _test_varargs
_test_varargs:
        .cfi_startproc
        sub        sp, #12
        push        {r4, r7, lr}
        add        r7, sp, #4
        .cfi_def_cfa r7, 20
        .cfi_offset lr, -16
        .cfi_offset r7, -20
        .cfi_offset r4, -24
        add.w        r9, r7, #8
        mov        r4, r0
        stm.w        r9, {r1, r2, r3}
        .cfi_endproc

@ CHECK-LABEL: start: {{.*}} _test_missing_lr
@ CHECK: compact encoding: 0x04000000
        .thumb_func _test_missing_lr
_test_missing_lr:
        .cfi_startproc
        push {r7}
        .cfi_def_cfa r7, 4
        .cfi_offset r7, -4
        pop {r7}
        bx lr
        .cfi_endproc

@ CHECK-LABEL: start: {{.*}} _test_swapped_offsets
@ CHECK: compact encoding: 0x04000000
        .thumb_func _test_swapped_offsets
_test_swapped_offsets:
        .cfi_startproc
        push {r7, lr}
        push {r10}
        push {r4}
        .cfi_def_cfa r7, 8
        .cfi_offset lr, -4
        .cfi_offset r7, -8
        .cfi_offset r10, -12
        .cfi_offset r4, -16
        pop {r4}
        pop {r10}
        pop {r7, pc}
        .cfi_endproc
