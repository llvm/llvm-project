; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

; rdar://problem/12848162

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16

;CHECK-LABEL: @example12(
; CHECK-LABEL: vector.body:
; CHECK: [[VEC_IND:%.+]] = phi <4 x i32>
; CHECK: store <4 x i32> [[VEC_IND]]
; CHECK: ret void
define void @example12() {
entry:
  br label %loop

loop:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %gep = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  %iv.trunc = trunc i64 %indvars.iv to i32
  store i32 %iv.trunc, i32* %gep, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

