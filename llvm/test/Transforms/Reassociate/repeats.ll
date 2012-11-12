; RUN: opt < %s -reassociate -S | FileCheck %s

; Tests involving repeated operations on the same value.

define i8 @nilpotent(i8 %x) {
; CHECK: @nilpotent
  %tmp = xor i8 %x, %x
  ret i8 %tmp
; CHECK: ret i8 0
}

define i2 @idempotent(i2 %x) {
; CHECK: @idempotent
  %tmp1 = and i2 %x, %x
  %tmp2 = and i2 %tmp1, %x
  %tmp3 = and i2 %tmp2, %x
  ret i2 %tmp3
; CHECK: ret i2 %x
}

define i2 @add(i2 %x) {
; CHECK: @add
  %tmp1 = add i2 %x, %x
  %tmp2 = add i2 %tmp1, %x
  %tmp3 = add i2 %tmp2, %x
  ret i2 %tmp3
; CHECK: ret i2 0
}

define i2 @cst_add() {
; CHECK: @cst_add
  %tmp1 = add i2 1, 1
  %tmp2 = add i2 %tmp1, 1
  ret i2 %tmp2
; CHECK: ret i2 -1
}

define i8 @cst_mul() {
; CHECK: @cst_mul
  %tmp1 = mul i8 3, 3
  %tmp2 = mul i8 %tmp1, 3
  %tmp3 = mul i8 %tmp2, 3
  %tmp4 = mul i8 %tmp3, 3
  ret i8 %tmp4
; CHECK: ret i8 -13
}

define i3 @foo3x5(i3 %x) {
; Can be done with two multiplies.
; CHECK: @foo3x5
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i3 %x, %x
  %tmp2 = mul i3 %tmp1, %x
  %tmp3 = mul i3 %tmp2, %x
  %tmp4 = mul i3 %tmp3, %x
  ret i3 %tmp4
}

define i3 @foo3x6(i3 %x) {
; Can be done with two multiplies.
; CHECK: @foo3x6
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i3 %x, %x
  %tmp2 = mul i3 %tmp1, %x
  %tmp3 = mul i3 %tmp2, %x
  %tmp4 = mul i3 %tmp3, %x
  %tmp5 = mul i3 %tmp4, %x
  ret i3 %tmp5
}

define i3 @foo3x7(i3 %x) {
; Can be done with two multiplies.
; CHECK: @foo3x7
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i3 %x, %x
  %tmp2 = mul i3 %tmp1, %x
  %tmp3 = mul i3 %tmp2, %x
  %tmp4 = mul i3 %tmp3, %x
  %tmp5 = mul i3 %tmp4, %x
  %tmp6 = mul i3 %tmp5, %x
  ret i3 %tmp6
}

define i4 @foo4x8(i4 %x) {
; Can be done with two multiplies.
; CHECK: @foo4x8
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  ret i4 %tmp7
}

define i4 @foo4x9(i4 %x) {
; Can be done with three multiplies.
; CHECK: @foo4x9
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  ret i4 %tmp8
}

define i4 @foo4x10(i4 %x) {
; Can be done with three multiplies.
; CHECK: @foo4x10
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  %tmp9 = mul i4 %tmp8, %x
  ret i4 %tmp9
}

define i4 @foo4x11(i4 %x) {
; Can be done with four multiplies.
; CHECK: @foo4x11
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  %tmp9 = mul i4 %tmp8, %x
  %tmp10 = mul i4 %tmp9, %x
  ret i4 %tmp10
}

define i4 @foo4x12(i4 %x) {
; Can be done with two multiplies.
; CHECK: @foo4x12
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  %tmp9 = mul i4 %tmp8, %x
  %tmp10 = mul i4 %tmp9, %x
  %tmp11 = mul i4 %tmp10, %x
  ret i4 %tmp11
}

define i4 @foo4x13(i4 %x) {
; Can be done with three multiplies.
; CHECK: @foo4x13
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  %tmp9 = mul i4 %tmp8, %x
  %tmp10 = mul i4 %tmp9, %x
  %tmp11 = mul i4 %tmp10, %x
  %tmp12 = mul i4 %tmp11, %x
  ret i4 %tmp12
}

define i4 @foo4x14(i4 %x) {
; Can be done with three multiplies.
; CHECK: @foo4x14
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  %tmp9 = mul i4 %tmp8, %x
  %tmp10 = mul i4 %tmp9, %x
  %tmp11 = mul i4 %tmp10, %x
  %tmp12 = mul i4 %tmp11, %x
  %tmp13 = mul i4 %tmp12, %x
  ret i4 %tmp13
}

define i4 @foo4x15(i4 %x) {
; Can be done with four multiplies.
; CHECK: @foo4x15
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret
  %tmp1 = mul i4 %x, %x
  %tmp2 = mul i4 %tmp1, %x
  %tmp3 = mul i4 %tmp2, %x
  %tmp4 = mul i4 %tmp3, %x
  %tmp5 = mul i4 %tmp4, %x
  %tmp6 = mul i4 %tmp5, %x
  %tmp7 = mul i4 %tmp6, %x
  %tmp8 = mul i4 %tmp7, %x
  %tmp9 = mul i4 %tmp8, %x
  %tmp10 = mul i4 %tmp9, %x
  %tmp11 = mul i4 %tmp10, %x
  %tmp12 = mul i4 %tmp11, %x
  %tmp13 = mul i4 %tmp12, %x
  %tmp14 = mul i4 %tmp13, %x
  ret i4 %tmp14
}
