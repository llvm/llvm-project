; RUN: opt -scalar-evolution-max-arith-depth=0 -scalar-evolution-max-cast-depth=0 -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

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

; Constant factors still get folded together.
define void @test_mul_consts(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: @test_mul_consts
; CHECK:       %s2 = mul i32 %s1, %p3
; CHECK-NEXT:  -->  (5040 * %a * %b * %c * %d * %e * %f)
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

; The outer *5 gets distributed because it is at depth=0, but the resulting
; nested multiply doesn't get flattened, because it is at depth=1.
define void @test_mul(i32 %a, i32 %b) {
; CHECK-LABEL: @test_mul
; CHECK:       %tmp2 = mul i32 %tmp1, 5
; CHECK-NEXT:  -->  (20 + (5 * (3 * %a)))
  %tmp0 = mul i32 %a, 3
  %tmp1 = add i32 %tmp0, 4
  %tmp2 = mul i32 %tmp1, 5
  ret void
}

define void @test_sext(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: @test_sext
; CHECK:        %se2 = sext i64 %iv2.inc to i128
; CHECK-NEXT:   -->  {(1 + (sext i64 {(sext i32 (1 + %a)<nsw> to i64),+,1}<nsw><%loop> to i128))<nsw>,+,1}<nsw><%loop2>
entry:
  br label %loop

loop:
  %iv = phi i32 [ %a, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nsw i32 %iv, 1
  %cond = icmp sle i32 %iv.inc, 50
  br i1 %cond, label %loop, label %between

between:
  %se = sext i32 %iv.inc to i64
  br label %loop2

loop2:
  %iv2 = phi i64 [ %se, %between ], [ %iv2.inc, %loop2 ]
  %iv2.inc = add nsw i64 %iv2, 1
  %cond2 = icmp sle i64 %iv2.inc, 50
  br i1 %cond2, label %loop2, label %exit

exit:
  %se2 = sext i64 %iv2.inc to i128
  ret void
}

define void @test_zext(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: @test_zext
; CHECK:          %ze2 = zext i64 %iv2.inc to i128
; CHECK-NEXT:     -->  {(1 + (zext i64 {7,+,1}<nuw><nsw><%loop> to i128))<nuw><nsw>,+,1}<nuw><nsw><%loop2>
entry:
  br label %loop

loop:
  %iv = phi i32 [ 6, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nsw i32 %iv, 1
  %cond = icmp sle i32 %iv.inc, 50
  br i1 %cond, label %loop, label %between

between:
  %ze = zext i32 %iv.inc to i64
  br label %loop2

loop2:
  %iv2 = phi i64 [ %ze, %between ], [ %iv2.inc, %loop2 ]
  %iv2.inc = add nuw i64 %iv2, 1
  %cond2 = icmp sle i64 %iv2.inc, 50
  br i1 %cond2, label %loop2, label %exit

exit:
  %ze2 = zext i64 %iv2.inc to i128
  ret void
}

define void @test_trunc(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: @test_trunc
; CHECK:          %trunc2 = trunc i64 %iv2.inc to i32
; CHECK-NEXT:     -->  {(trunc i64 (1 + {7,+,1}<%loop>)<nuw><nsw> to i32),+,1}<%loop2> U: [8,53) S: [8,53)  -->  52 U: [52,53) S: [52,53)
entry:
  br label %loop

loop:
  %iv = phi i128 [ 6, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nsw i128 %iv, 1
  %cond = icmp sle i128 %iv.inc, 50
  br i1 %cond, label %loop, label %between

between:
  %trunc = trunc i128 %iv.inc to i64
  br label %loop2

loop2:
  %iv2 = phi i64 [ %trunc, %between ], [ %iv2.inc, %loop2 ]
  %iv2.inc = add nuw i64 %iv2, 1
  %cond2 = icmp sle i64 %iv2.inc, 50
  br i1 %cond2, label %loop2, label %exit

exit:
  %trunc2 = trunc i64 %iv2.inc to i32
  ret void
}

; Check that all constant SCEVs are folded regardless depth limit.
define void @test_mul_const(i32 %a) {
; CHECK-LABEL:  @test_mul_const
; CHECK:          %test3 = mul i32 %test2, 3
; CHECK-NEXT:     -->  (9 + (3 * (3 * %a)))
; CHECK:          %test4 = mul i32 3, 3
; CHECK-NEXT:     -->  9 U: [9,10) S: [9,10)
  %test = mul i32 3, %a
  %test2 = add i32 3, %test
  %test3 = mul i32 %test2, 3
  %test4 = mul i32 3, 3
  ret void
}
