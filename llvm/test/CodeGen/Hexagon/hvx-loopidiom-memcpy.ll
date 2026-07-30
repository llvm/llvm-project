; RUN: opt -march=hexagon -hexagon-loop-idiom -S < %s | FileCheck %s
; RUN: opt -mtriple=hexagon-- -p hexagon-loop-idiom -disable-memcpy-idiom -S < %s | FileCheck %s

; Make sure we don't convert load/store loops into memcpy if the access type
; is a vector. Using vector instructions is generally better in such cases.

; CHECK-NOT: @llvm.memcpy

%s.0 = type { i32 }

define void @f0(ptr noalias %a0, ptr noalias %a1) #0 align 2 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v7, %b1 ], [ 0, %b0 ]
  %v1 = mul nuw nsw i32 %v0, 64
  %v2 = getelementptr %s.0, ptr %a0, i32 %v1
  %v3 = getelementptr %s.0, ptr %a1, i32 %v1
  %v5 = load <64 x i32>, ptr %v2, align 256
  store <64 x i32> %v5, ptr %v3, align 256
  %v7 = add nuw nsw i32 %v0, 1
  br i1 undef, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
