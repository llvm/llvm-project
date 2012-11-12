@ RUN: llvm-mc -triple armv7-unknown-unknown %s --show-encoding > %t
@ RUN: FileCheck < %t %s

    bl _printf
@ CHECK: bl _printf @ encoding: [A,A,A,0xeb]
@ CHECK: @ fixup A - offset: 0, value: _printf, kind: fixup_arm_uncondbl

    mov r9, :lower16:(_foo)
    movw r9, :lower16:(_foo)
    movt r9, :upper16:(_foo)

@ CHECK: movw	r9, :lower16:_foo       @ encoding: [A,0x90'A',0b0000AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movw_lo16
@ CHECK: movw	r9, :lower16:_foo       @ encoding: [A,0x90'A',0b0000AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movw_lo16
@ CHECK: movt	r9, :upper16:_foo       @ encoding: [A,0x90'A',0b0100AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movt_hi16

    mov r2, fred

@ CHECK: movw  r2, fred                 @ encoding: [A,0x20'A',0b0000AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: fred, kind: fixup_arm_movw_lo16
