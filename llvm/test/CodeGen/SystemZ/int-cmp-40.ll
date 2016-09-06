; Test 64-bit comparisons in which the second operand is zero-extended
; from a PC-relative i16.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i16 1
@h = global i16 1, align 1, section "foo"

; Check unsigned comparison.
define i64 @f1(i64 %src1) {
; CHECK-LABEL: f1:
; CHECK: clghrl %r2, g
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i16 , i16 *@g
  %src2 = zext i16 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %tmp = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  %res = add i64 %tmp, 1
  ret i64 %res
}

; Check signed comparison.
define i64 @f2(i64 %src1) {
; CHECK-LABEL: f2:
; CHECK-NOT: clghrl
; CHECK: br %r14
entry:
  %val = load i16 , i16 *@g
  %src2 = zext i16 %val to i64
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %tmp = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  %res = add i64 %tmp, 1
  ret i64 %res
}

; Check equality.
define i64 @f3(i64 %src1) {
; CHECK-LABEL: f3:
; CHECK: clghrl %r2, g
; CHECK-NEXT: je
; CHECK: br %r14
entry:
  %val = load i16 , i16 *@g
  %src2 = zext i16 %val to i64
  %cond = icmp eq i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %tmp = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  %res = add i64 %tmp, 1
  ret i64 %res
}

; Check inequality.
define i64 @f4(i64 %src1) {
; CHECK-LABEL: f4:
; CHECK: clghrl %r2, g
; CHECK-NEXT: jlh
; CHECK: br %r14
entry:
  %val = load i16 , i16 *@g
  %src2 = zext i16 %val to i64
  %cond = icmp ne i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %tmp = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  %res = add i64 %tmp, 1
  ret i64 %res
}

; Repeat f1 with an unaligned address.
define i64 @f5(i64 %src1) {
; CHECK-LABEL: f5:
; CHECK: lgrl [[REG:%r[0-5]]], h@GOT
; CHECK: llgh [[VAL:%r[0-5]]], 0([[REG]])
; CHECK: clgrjl %r2, [[VAL]],
; CHECK: br %r14
entry:
  %val = load i16 , i16 *@h, align 1
  %src2 = zext i16 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %tmp = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  %res = add i64 %tmp, 1
  ret i64 %res
}

; Check the comparison can be reversed if that allows CLGHRL to be used.
define i64 @f6(i64 %src2) {
; CHECK-LABEL: f6:
; CHECK: clghrl %r2, g
; CHECK-NEXT: jh {{\.L.*}}
; CHECK: br %r14
entry:
  %val = load i16 , i16 *@g
  %src1 = zext i16 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src2, %src2
  br label %exit
exit:
  %tmp = phi i64 [ %src2, %entry ], [ %mul, %mulb ]
  %res = add i64 %tmp, 1
  ret i64 %res
}
