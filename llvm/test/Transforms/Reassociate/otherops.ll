; Reassociation should apply to Add, Mul, And, Or, & Xor
;
; RUN: opt < %s -reassociate -instcombine -die -S | FileCheck %s

define i32 @test_mul(i32 %arg) {
; CHECK-LABEL: test_mul
; CHECK-NEXT: %tmp2 = mul i32 %arg, 144
; CHECK-NEXT: ret i32 %tmp2

  %tmp1 = mul i32 12, %arg
  %tmp2 = mul i32 %tmp1, 12
  ret i32 %tmp2
}

define i32 @test_and(i32 %arg) {
; CHECK-LABEL: test_and
; CHECK-NEXT: %tmp2 = and i32 %arg, 14
; CHECK-NEXT: ret i32 %tmp2

  %tmp1 = and i32 14, %arg
  %tmp2 = and i32 %tmp1, 14
  ret i32 %tmp2
}

define i32 @test_or(i32 %arg) {
; CHECK-LABEL: test_or
; CHECK-NEXT: %tmp2 = or i32 %arg, 14
; CHECK-NEXT: ret i32 %tmp2

  %tmp1 = or i32 14, %arg
  %tmp2 = or i32 %tmp1, 14
  ret i32 %tmp2
}

define i32 @test_xor(i32 %arg) {
; CHECK-LABEL: test_xor
; CHECK-NEXT: ret i32 %arg

  %tmp1 = xor i32 12, %arg
  %tmp2 = xor i32 %tmp1, 12
  ret i32 %tmp2
}
