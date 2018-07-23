; Test 64-bit comparisons in which the second operand is a PC-relative
; variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i64 1
@h = global i64 1, align 4, section "foo"

; Check signed comparisons.
define i64 @f1(i64 %src1) {
; CHECK-LABEL: f1:
; CHECK: cgrl %r2, g
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i64, i64 *@g
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check unsigned comparisons.
define i64 @f2(i64 %src1) {
; CHECK-LABEL: f2:
; CHECK: clgrl %r2, g
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i64, i64 *@g
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check equality, which can use CRL or CLRL.
define i64 @f3(i64 %src1) {
; CHECK-LABEL: f3:
; CHECK: c{{l?}}grl %r2, g
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %src2 = load i64, i64 *@g
  %cond = icmp eq i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; ...likewise inequality.
define i64 @f4(i64 %src1) {
; CHECK-LABEL: f4:
; CHECK: c{{l?}}grl %r2, g
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %src2 = load i64, i64 *@g
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
define i64 @f5(i64 %src1) {
; CHECK-LABEL: f5:
; CHECK: larl [[REG:%r[0-5]]], h
; CHECK: cg %r2, 0([[REG]])
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i64, i64 *@h, align 4
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check the comparison can be reversed if that allows CGRL to be used.
define i64 @f6(i64 %src2) {
; CHECK-LABEL: f6:
; CHECK: cgrl %r2, g
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %src1 = load i64, i64 *@g
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src2, %src2
  br label %exit
exit:
  %res = phi i64 [ %src2, %entry ], [ %mul, %mulb ]
  ret i64 %res
}
