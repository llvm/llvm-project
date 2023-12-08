; RUN: opt < %s -S -passes=loop-vectorize | FileCheck %s

; Check that we can vectorize this loop without crashing.

target triple = "aarch64-none-linux-gnu"
define float @quux() {
; CHECK: @quux
bb:
  br label %bb1

bb1:
  %tmp = phi i64 [ %tmp3, %bb1 ], [ 0, %bb ]
  %tmp2 = phi float [ %tmp5, %bb1 ], [ 0.000000e+00, %bb ]
  %tmp3 = add nsw i64 %tmp, 1
  %tmp5 = fadd float %tmp2, 3.000000e+00
  %tmp6 = mul i32 0, 0
  %tmp7 = icmp sgt i64 %tmp, 0
  br i1 %tmp7, label %bb8, label %bb1

bb8:
  %tmp9 = phi float [ %tmp5, %bb1 ]
  ret float %tmp9
}