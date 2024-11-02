; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-linux-gnu"

%pair = type { ptr, ptr }

define void @test_pr55375_interleave_opaque_ptr(ptr %start, ptr %end) {
; CHECK-LABEL: @test_pr55375_interleave_opaque_ptr(
; CHECK:       vector.body:
; CHECK-NEXT:    [[POINTER_PHI:%.*]] = phi ptr [ %start, %vector.ph ], [ [[PTR_IND:%.*]], %vector.body ]
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[POINTER_PHI]], <2 x i64> <i64 0, i64 16>
; CHECK-NEXT:    [[TMP7:%.*]] = extractelement <2 x ptr> [[TMP5]], i32 0
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr ptr, ptr [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP11:%.*]] = shufflevector <2 x ptr> zeroinitializer, <2 x ptr> [[TMP5]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[INTERLEAVED_VEC:%.*]] = shufflevector <4 x ptr> [[TMP11]], <4 x ptr> poison, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
; CHECK-NEXT:    store <4 x ptr> [[INTERLEAVED_VEC]], ptr [[TMP8]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[PTR_IND]] = getelementptr i8, ptr [[POINTER_PHI]], i64 32
; CHECK-NEXT:    [[TMP13:%.*]] = icmp eq i64 [[INDEX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[TMP13]], label %middle.block, label %vector.body
;
entry:
  br label %loop

loop:
  %iv = phi ptr [ %start, %entry ], [ %iv.next, %loop ]
  %iv.1 = getelementptr inbounds %pair, ptr %iv, i64 0, i32 1
  store ptr %iv, ptr %iv.1, align 8
  store ptr null, ptr %iv, align 8
  %iv.next = getelementptr inbounds %pair, ptr %iv, i64 1
  %ec = icmp eq ptr %iv.next, %end
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
