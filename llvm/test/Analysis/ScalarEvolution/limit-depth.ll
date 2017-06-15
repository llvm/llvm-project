; RUN: opt -scalar-evolution-max-arith-depth=0 -analyze -scalar-evolution < %s | FileCheck %s

; Check that depth set to 0 prevents getAddExpr and getMulExpr from making
; transformations in SCEV. We expect the result to be very straightforward.

define void @test_add(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: @test_add
; CHECK:       %s2 = add i32 %s1, %p3
; CHECK-NEXT:   -->  (%a + %a + %b + %b + %c + %c + %d + %d + %e + %e + %f + %f)
  %tmp0 = add i32 %a, %b
  %tmp1 = add i32 %b, %c
  %tmp2 = add i32 %c, %d
  %tmp3 = add i32 %d, %e
  %tmp4 = add i32 %e, %f
  %tmp5 = add i32 %f, %a

  %p1 = add i32 %tmp0, %tmp3
  %p2 = add i32 %tmp1, %tmp4
  %p3 = add i32 %tmp2, %tmp5

  %s1 = add i32 %p1, %p2
  %s2 = add i32 %s1, %p3
  ret void
}

define void @test_mul(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: @test_mul
; CHECK:       %s2 = mul i32 %s1, %p3
; CHECK-NEXT:  -->  (2 * 3 * 4 * 5 * 6 * 7 * %a * %b * %c * %d * %e * %f)
  %tmp0 = mul i32 %a, 2
  %tmp1 = mul i32 %b, 3
  %tmp2 = mul i32 %c, 4
  %tmp3 = mul i32 %d, 5
  %tmp4 = mul i32 %e, 6
  %tmp5 = mul i32 %f, 7

  %p1 = mul i32 %tmp0, %tmp3
  %p2 = mul i32 %tmp1, %tmp4
  %p3 = mul i32 %tmp2, %tmp5

  %s1 = mul i32 %p1, %p2
  %s2 = mul i32 %s1, %p3
  ret void
}
