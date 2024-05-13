; Test 64-bit comparisons in which the second operand is zero-extended
; from a PC-relative i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = dso_local global i32 1
@h = dso_local global i32 1, align 2, section "foo"

; Check unsigned comparison.
define dso_local i64 @f1(i64 %src1) {
; CHECK-LABEL: f1:
; CHECK: clgfrl %r2, g
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %val = load i32, ptr@g
  %src2 = zext i32 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check signed comparison.
define dso_local i64 @f2(i64 %src1) {
; CHECK-LABEL: f2:
; CHECK-NOT: clgfrl
; CHECK: br %r14
entry:
  %val = load i32, ptr@g
  %src2 = zext i32 %val to i64
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check equality.
define dso_local i64 @f3(i64 %src1) {
; CHECK-LABEL: f3:
; CHECK: clgfrl %r2, g
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %val = load i32, ptr@g
  %src2 = zext i32 %val to i64
  %cond = icmp eq i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check inequality.
define dso_local i64 @f4(i64 %src1) {
; CHECK-LABEL: f4:
; CHECK: clgfrl %r2, g
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %val = load i32, ptr@g
  %src2 = zext i32 %val to i64
  %cond = icmp ne i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Repeat f1 with an unaligned address.
define dso_local i64 @f5(i64 %src1) {
; CHECK-LABEL: f5:
; CHECK: larl [[REG:%r[0-5]]], h
; CHECK: clgf %r2, 0([[REG]])
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %val = load i32, ptr@h, align 2
  %src2 = zext i32 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check the comparison can be reversed if that allows CLGFRL to be used.
define dso_local i64 @f6(i64 %src2) {
; CHECK-LABEL: f6:
; CHECK: clgfrl %r2, g
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %val = load i32, ptr@g
  %src1 = zext i32 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src2, %src2
  br label %exit
exit:
  %res = phi i64 [ %src2, %entry ], [ %mul, %mulb ]
  ret i64 %res
}
