; Test 32-bit comparisons in which the second operand is a PC-relative
; variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i32 1
@h = global i32 1, align 2, section "foo"

; Check signed comparisons.
define i32 @f1(i32 %src1) {
; CHECK-LABEL: f1:
; CHECK: crl %r2, g
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i32, i32 *@g
  %cond = icmp slt i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check unsigned comparisons.
define i32 @f2(i32 %src1) {
; CHECK-LABEL: f2:
; CHECK: clrl %r2, g
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i32, i32 *@g
  %cond = icmp ult i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check equality, which can use CRL or CLRL.
define i32 @f3(i32 %src1) {
; CHECK-LABEL: f3:
; CHECK: c{{l?}}rl %r2, g
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %src2 = load i32, i32 *@g
  %cond = icmp eq i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; ...likewise inequality.
define i32 @f4(i32 %src1) {
; CHECK-LABEL: f4:
; CHECK: c{{l?}}rl %r2, g
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %src2 = load i32, i32 *@g
  %cond = icmp ne i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Repeat f1 with an unaligned address.
define i32 @f5(i32 %src1) {
; CHECK-LABEL: f5:
; CHECK: larl [[REG:%r[0-5]]], h
; CHECK: c %r2, 0([[REG]])
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i32, i32 *@h, align 2
  %cond = icmp slt i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Repeat f2 with an unaligned address.
define i32 @f6(i32 %src1) {
; CHECK-LABEL: f6:
; CHECK: larl [[REG:%r[0-5]]], h
; CHECK: cl %r2, 0([[REG]])
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %src2 = load i32, i32 *@h, align 2
  %cond = icmp ult i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check the comparison can be reversed if that allows CRL to be used.
define i32 @f7(i32 %src2) {
; CHECK-LABEL: f7:
; CHECK: crl %r2, g
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %src1 = load i32, i32 *@g
  %cond = icmp slt i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src2, %src2
  br label %exit
exit:
  %res = phi i32 [ %src2, %entry ], [ %mul, %mulb ]
  ret i32 %res
}
