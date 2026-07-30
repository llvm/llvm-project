@ RUN: llvm-mc -triple armv6m-unknown-unknown %s --show-encoding -o - | \
@ RUN:   FileCheck %s

    movs r3, :upper8_15:_foo
    adds r3, :upper0_7:_foo
    adds r3, :lower8_15:_foo
    adds r3, :lower0_7:_foo

@ CHECK:      movs    r3, :upper8_15:_foo             @ encoding: [A,0x23]
@ CHECK-NEXT: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_upper_8_15
@ CHECK-NEXT: adds    r3, :upper0_7:_foo              @ encoding: [A,0x33]
@ CHECK-NEXT: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_upper_0_7
@ CHECK-NEXT: adds    r3, :lower8_15:_foo             @ encoding: [A,0x33]
@ CHECK-NEXT: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_lower_8_15
@ CHECK-NEXT: adds    r3, :lower0_7:_foo              @ encoding: [A,0x33]
@ CHECK-NEXT: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_lower_0_7

@ GNU syntax variants:
    movs r3, #:upper8_15:#_foo
    movs r3, #:upper8_15:_foo

@ CHECK:      movs    r3, :upper8_15:_foo             @ encoding: [A,0x23]
@ CHECK-NEXT: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_upper_8_15
@ CHECK-NEXT: movs    r3, :upper8_15:_foo             @ encoding: [A,0x23]
@ CHECK-NEXT: @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_upper_8_15
