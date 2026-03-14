@ RUN: llvm-mc -triple armv7-unknown-unknown %s --show-encoding > %t
@ RUN: FileCheck < %t %s
@ RUN: llvm-mc -triple armebv7-unknown-unknown %s --show-encoding > %t
@ RUN: FileCheck --check-prefix=CHECK-BE < %t %s

    bl _printf
@ CHECK: bl _printf @ encoding: [A,A,A,0xeb]
@ CHECK: @ fixup A - offset: 0, value: _printf, kind: fixup_arm_uncondbl
@ CHECK-BE: bl _printf @ encoding: [0xeb,A,A,A]
@ CHECK-BE: @ fixup A - offset: 0, value: _printf, kind: fixup_arm_uncondbl

    mov r9, :lower16:(_foo)
    movw r9, :lower16:(_foo)
    movt r9, :upper16:(_foo)

@ CHECK: movw	r9, :lower16:_foo       @ encoding: [A,0x90'A',0b0000AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movw_lo16
@ CHECK-BE: movw	r9, :lower16:_foo       @ encoding: [0xe3,0b0000AAAA,0x90'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movw_lo16
@ CHECK: movw	r9, :lower16:_foo       @ encoding: [A,0x90'A',0b0000AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movw_lo16
@ CHECK-BE: movw	r9, :lower16:_foo       @ encoding: [0xe3,0b0000AAAA,0x90'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movw_lo16
@ CHECK: movt	r9, :upper16:_foo       @ encoding: [A,0x90'A',0b0100AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movt_hi16
@ CHECK-BE: movt	r9, :upper16:_foo       @ encoding: [0xe3,0b0100AAAA,0x90'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_movt_hi16

    mov r2, :lower16:fred

@ CHECK: movw  r2, :lower16:fred                 @ encoding: [A,0x20'A',0b0000AAAA,0xe3]
@ CHECK: @   fixup A - offset: 0, value: fred, kind: fixup_arm_movw_lo16
@ CHECK-BE: movw  r2, :lower16:fred                 @ encoding: [0xe3,0b0000AAAA,0x20'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: fred, kind: fixup_arm_movw_lo16

    add r0, r0, #(L1 - L2)

@ CHECK: add     r0, r0, #L1-L2          @ encoding: [A,0b0000AAAA,0x80,0xe2]
@ CHECK: @   fixup A - offset: 0, value: L1-L2, kind: fixup_arm_mod_imm
@ CHECK-BE: add     r0, r0, #L1-L2          @ encoding: [0xe2,0x80,0b0000AAAA,A]
@ CHECK-BE: @   fixup A - offset: 0, value: L1-L2, kind: fixup_arm_mod_imm
