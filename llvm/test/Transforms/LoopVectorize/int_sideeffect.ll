; RUN: opt -S < %s -passes=loop-vectorize -force-vector-width=4 | FileCheck %s

declare void @llvm.sideeffect()

; Vectorization across a @llvm.sideeffect.

; CHECK-LABEL: store_ones
; CHECK: store <4 x float>
define void @store_ones(float* %p, i64 %n) nounwind {
bb7.lr.ph:
  br label %bb7

bb7:
  %i.02 = phi i64 [ 0, %bb7.lr.ph ], [ %tmp13, %bb7 ]
  call void @llvm.sideeffect()
  %tmp10 = getelementptr inbounds float, float* %p, i64 %i.02
  store float 1.0, float* %tmp10, align 4
  %tmp13 = add i64 %i.02, 1
  %tmp6 = icmp ult i64 %tmp13, %n
  br i1 %tmp6, label %bb7, label %bb14

bb14:
  ret void
}
