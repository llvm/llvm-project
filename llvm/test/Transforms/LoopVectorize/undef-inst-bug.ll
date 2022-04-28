; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; We use to fail on this loop because we did not properly handle the loop
; invariant instruction anchored in the loop when used as a getelementptr index.
; We would use the index from the original loop resulting in a use not dominated
; by the definition.

; PR16452

; Verify that we don't miscompile this loop.

; CHECK-LABEL: @t(
; CHECK: <4 x i32>

define void @t() {
entry:
  br label %for.body

for.body:
  %indvars.iv17 = phi i64 [ %indvars.next, %for.body ], [ 128, %entry ]

  ; Loop invariant anchored in loop.
  %idxprom21 = zext i32 undef to i64

  %arrayidx23 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* undef, i64 0, i64 %idxprom21, i64 %indvars.iv17
  store i32 undef, i32* %arrayidx23, align 4
  %indvars.next= add i64 %indvars.iv17, -1
  %0 = trunc i64 %indvars.next to i32
  %cmp15 = icmp ugt i32 %0, undef
  br i1 %cmp15, label %for.body, label %loopexit

loopexit:
  ret void
}
