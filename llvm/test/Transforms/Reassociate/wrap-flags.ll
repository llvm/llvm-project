; RUN: opt < %s -reassociate -dce -S | FileCheck %s
; PR12985

; Verify the nsw flags are preserved when converting shl to mul.

; CHECK-LABEL: @shl_to_mul_nsw(
; CHECK: %mul = mul i32 %i, -2147483648
; CHECK: add i32 %mul, 1
define i32 @shl_to_mul_nsw(i32 %i) {
entry:
  %mul = shl nsw i32 %i, 31
  %mul2 = add i32 %mul, 1
  ret i32 %mul2
}

; CHECK-LABEL: @shl_to_mul_nuw(
; CHECK: %mul = mul nuw i32 %i, 4
; CHECK: add i32 %mul, 1
define i32 @shl_to_mul_nuw(i32 %i) {
entry:
  %mul = shl nuw i32 %i, 2
  %mul2 = add i32 %mul, 1
  ret i32 %mul2
}

; CHECK-LABEL: @shl_to_mul_nuw_nsw(
; CHECK: %mul = mul nuw nsw i32 %i, 4
; CHECK: add i32 %mul, 1
define i32 @shl_to_mul_nuw_nsw(i32 %i) {
entry:
  %mul = shl nuw nsw i32 %i, 2
  %mul2 = add i32 %mul, 1
  ret i32 %mul2
}

; CHECK-LABEL: @pr23926(
; CHECK:       %[[X1_neg:.*]] = sub i2 0, %X1
; CHECK-NEXT:  %[[sub_one:.*]] = add i2 %[[X1_neg]], -1
; CHECK-NEXT:  %[[add:.*]] = add i2 %[[sub_one]], %X2
; CHECK-NEXT:  ret i2 %[[add]]
define i2 @pr23926(i2 %X1, i2 %X2) {
  %add = add nuw i2 %X1, 1
  %sub = sub nuw nsw i2 %X2, %add
  ret i2 %sub
}
