; RUN: opt -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@uf = common global [100 x i32] zeroinitializer, align 16
@xi = common global [100 x i32] zeroinitializer, align 16
@q = common global [100 x i32] zeroinitializer, align 16

; PR16455


; Due to a bug in the way we handled reverse induction stores we would generate
; a shuffle too many.

define void @t()  {
entry:
  br label %for.body

; CHECK-LABEL: @t(
; CHECK: vector.body:
; CHECK: [[VAR1:%[a-zA-Z0-9.]+]] = load <4 x i32>
; CHECK: [[VAR2:%[a-zA-Z0-9.]+]] = load <4 x i32>
; CHECK: [[VAR3:%[a-zA-Z0-9]+]] = add nsw <4 x i32> [[VAR2]], [[VAR1]]
; CHECK: store <4 x i32> [[VAR3]]
; CHECK: [[VAR4:%[a-zA-Z0-9.]+]] = load <4 x i32>
; CHECK: add nsw <4 x i32> [[VAR3]], [[VAR4]]
; CHECK-NOT: shufflevector

for.body:
  %indvars.iv = phi i64 [ 93, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = add i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @uf, i64 0, i64 %0
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* @xi, i64 0, i64 %0
  %1 = load i32, i32* %arrayidx3, align 4
  %2 = load i32, i32* %arrayidx, align 4
  %add4 = add nsw i32 %2, %1
  store i32 %add4, i32* %arrayidx, align 4
  %arrayidx7 = getelementptr inbounds [100 x i32], [100 x i32]* @q, i64 0, i64 %0
  %3 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %add4, %3
  store i32 %add8, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, -1
  %4 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp ugt i32 %4, 2
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}
