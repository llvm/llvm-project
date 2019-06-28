; RUN: llc --verify-machineinstrs -mtriple=thumbv8.1-m.main-none-eabi -mattr=+mve %s -o - | FileCheck %s -check-prefix=CHECK --check-prefix=CHECK-MVE
; RUN: llc --verify-machineinstrs -mtriple=thumbv8.1-m.main-none-eabi %s -o - | FileCheck %s -check-prefix=CHECK --check-prefix=CHECK-NON-MVE

define i64 @shift_left_reg(i64 %x, i64 %y) {
; CHECK-MVE-LABEL: shift_left_reg:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    lsll r0, r1, r2
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_left_reg:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    .save {r7, lr}
; CHECK-NON-MVE-NEXT:    push {r7, lr}
; CHECK-NON-MVE-NEXT:    bl __aeabi_llsl
; CHECK-NON-MVE-NEXT:    pop {r7}
; CHECK-NON-MVE-NEXT:    pop {r2}
; CHECK-NON-MVE-NEXT:    bx r2
entry:
  %shl = shl i64 %x, %y
  ret i64 %shl
}

define i64 @shift_left_imm(i64 %x) {
; CHECK-MVE-LABEL: shift_left_imm:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    lsll r0, r1, #3
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_left_imm:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    lsrs r2, r0, #29
; CHECK-NON-MVE-NEXT:    lsls r1, r1, #3
; CHECK-NON-MVE-NEXT:    adds r1, r1, r2
; CHECK-NON-MVE-NEXT:    lsls r0, r0, #3
; CHECK-NON-MVE-NEXT:    bx lr
entry:
  %shl = shl i64 %x, 3
  ret i64 %shl
}

define i64 @shift_left_imm_big(i64 %x) {
; CHECK-LABEL: shift_left_imm_big:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    lsls r1, r0, #16
; CHECK-NEXT:    movs r0, #0
; CHECK-NEXT:    bx lr
entry:
  %shl = shl i64 %x, 48
  ret i64 %shl
}

define i64 @shift_left_imm_big2(i64 %x) {
; CHECK-MVE-LABEL: shift_left_imm_big2:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    lsll r0, r1, #32
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_left_imm_big2:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    movs r1, r0
; CHECK-NON-MVE-NEXT:    movs r0, #0
; CHECK-NON-MVE-NEXT:    bx lr
entry:
  %shl = shl i64 %x, 32
  ret i64 %shl
}

define i64 @shift_left_imm_big3(i64 %x) {
; CHECK-LABEL: shift_left_imm_big3:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    lsls r1, r0, #1
; CHECK-NEXT:    movs r0, #0
; CHECK-NEXT:    bx lr
entry:
  %shl = shl i64 %x, 33
  ret i64 %shl
}

define i64 @shift_right_reg(i64 %x, i64 %y) {
; CHECK-MVE-LABEL: shift_right_reg:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    rsbs r2, r2, #0
; CHECK-MVE-NEXT:    lsll r0, r1, r2
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_right_reg:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    .save {r7, lr}
; CHECK-NON-MVE-NEXT:    push {r7, lr}
; CHECK-NON-MVE-NEXT:    bl __aeabi_llsr
; CHECK-NON-MVE-NEXT:    pop {r7}
; CHECK-NON-MVE-NEXT:    pop {r2}
; CHECK-NON-MVE-NEXT:    bx r2
entry:
  %shr = lshr i64 %x, %y
  ret i64 %shr
}

define i64 @shift_right_imm(i64 %x) {
; CHECK-MVE-LABEL: shift_right_imm:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    lsrl r0, r1, #3
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_right_imm:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    lsls r2, r1, #29
; CHECK-NON-MVE-NEXT:    lsrs r0, r0, #3
; CHECK-NON-MVE-NEXT:    adds r0, r0, r2
; CHECK-NON-MVE-NEXT:    lsrs r1, r1, #3
; CHECK-NON-MVE-NEXT:    bx lr
entry:
  %shr = lshr i64 %x, 3
  ret i64 %shr
}

define i64 @shift_right_imm_big(i64 %x) {
; CHECK-LABEL: shift_right_imm_big:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    lsrs r0, r1, #16
; CHECK-NEXT:    movs r1, #0
; CHECK-NEXT:    bx lr
entry:
  %shr = lshr i64 %x, 48
  ret i64 %shr
}

define i64 @shift_right_imm_big2(i64 %x) {
; CHECK-MVE-LABEL: shift_right_imm_big2:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    lsrl r0, r1, #32
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_right_imm_big2:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    movs r0, r1
; CHECK-NON-MVE-NEXT:    movs r1, #0
; CHECK-NON-MVE-NEXT:    bx lr
entry:
  %shr = lshr i64 %x, 32
  ret i64 %shr
}

define i64 @shift_right_imm_big3(i64 %x) {
; CHECK-LABEL: shift_right_imm_big3:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    lsrs r0, r1, #1
; CHECK-NEXT:    movs r1, #0
; CHECK-NEXT:    bx lr
entry:
  %shr = lshr i64 %x, 33
  ret i64 %shr
}

define i64 @shift_arithmetic_right_reg(i64 %x, i64 %y) {
; CHECK-MVE-LABEL: shift_arithmetic_right_reg:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    asrl r0, r1, r2
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_arithmetic_right_reg:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    .save {r7, lr}
; CHECK-NON-MVE-NEXT:    push {r7, lr}
; CHECK-NON-MVE-NEXT:    bl __aeabi_lasr
; CHECK-NON-MVE-NEXT:    pop {r7}
; CHECK-NON-MVE-NEXT:    pop {r2}
; CHECK-NON-MVE-NEXT:    bx r2
entry:
  %shr = ashr i64 %x, %y
  ret i64 %shr
}

define i64 @shift_arithmetic_right_imm(i64 %x) {
; CHECK-MVE-LABEL: shift_arithmetic_right_imm:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    asrl r0, r1, #3
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: shift_arithmetic_right_imm:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    lsls r2, r1, #29
; CHECK-NON-MVE-NEXT:    lsrs r0, r0, #3
; CHECK-NON-MVE-NEXT:    adds r0, r0, r2
; CHECK-NON-MVE-NEXT:    asrs r1, r1, #3
; CHECK-NON-MVE-NEXT:    bx lr
entry:
  %shr = ashr i64 %x, 3
  ret i64 %shr
}

%struct.bar = type { i16, i8, [5 x i8] }

define arm_aapcs_vfpcc void @fn1(%struct.bar* nocapture %a) {
; CHECK-MVE-LABEL: fn1:
; CHECK-MVE:       @ %bb.0: @ %entry
; CHECK-MVE-NEXT:    ldr r2, [r0, #4]
; CHECK-MVE-NEXT:    movs r1, #0
; CHECK-MVE-NEXT:    lsll r2, r1, #8
; CHECK-MVE-NEXT:    strb r1, [r0, #7]
; CHECK-MVE-NEXT:    str.w r2, [r0, #3]
; CHECK-MVE-NEXT:    bx lr
;
; CHECK-NON-MVE-LABEL: fn1:
; CHECK-NON-MVE:       @ %bb.0: @ %entry
; CHECK-NON-MVE-NEXT:    ldr r1, [r0, #4]
; CHECK-NON-MVE-NEXT:    lsls r2, r1, #8
; CHECK-NON-MVE-NEXT:    movs r3, #3
; CHECK-NON-MVE-NEXT:    str r2, [r0, r3]
; CHECK-NON-MVE-NEXT:    adds r0, r0, #3
; CHECK-NON-MVE-NEXT:    lsrs r1, r1, #24
; CHECK-NON-MVE-NEXT:    strb r1, [r0, #4]
; CHECK-NON-MVE-NEXT:    bx lr
entry:
  %carey = getelementptr inbounds %struct.bar, %struct.bar* %a, i32 0, i32 2
  %0 = bitcast [5 x i8]* %carey to i40*
  %bf.load = load i40, i40* %0, align 1
  %bf.clear = and i40 %bf.load, -256
  store i40 %bf.clear, i40* %0, align 1
  ret void
}
