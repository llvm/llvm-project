; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-ios %s -S -debug -disable-output 2>&1 | FileCheck --check-prefix=CM %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 %s -S | FileCheck --check-prefix=FORCED %s

; Test case from PR41294.

; Check scalar cost for extractvalue. The constant and loop invariant operands are free,
; leaving cost 3 for scalarizing the result + 2 for executing the op with VF 2.

; CM: LV: Found uniform instruction:   %a = extractvalue { i64, i64 } %sv, 0
; CM: LV: Found uniform instruction:   %b = extractvalue { i64, i64 } %sv, 1

; CM: LV: Scalar loop costs: 5.
; CM: LV: Found an estimated cost of 0 for VF 2 For instruction:   %a = extractvalue { i64, i64 } %sv, 0
; CM-NEXT: LV: Found an estimated cost of 0 for VF 2 For instruction:   %b = extractvalue { i64, i64 } %sv, 1

; Check that the extractvalue operands are actually free in vector code.

; FORCED-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; FORCED-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; FORCED-NEXT:    %0 = add i32 %index, 0
; FORCED-NEXT:    %1 = extractvalue { i64, i64 } %sv, 0
; FORCED-NEXT:    %broadcast.splatinsert = insertelement <2 x i64> poison, i64 %1, i64 0
; FORCED-NEXT:    %broadcast.splat = shufflevector <2 x i64> %broadcast.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    %2 = extractvalue { i64, i64 } %sv, 1
; FORCED-NEXT:    %broadcast.splatinsert1 = insertelement <2 x i64> poison, i64 %2, i64 0
; FORCED-NEXT:    %broadcast.splat2 = shufflevector <2 x i64> %broadcast.splatinsert1, <2 x i64> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    %3 = getelementptr i64, ptr %dst, i32 %0
; FORCED-NEXT:    %4 = add <2 x i64> %broadcast.splat, %broadcast.splat2
; FORCED-NEXT:    %5 = getelementptr i64, ptr %3, i32 0
; FORCED-NEXT:    store <2 x i64> %4, ptr %5, align 4
; FORCED-NEXT:    %index.next = add nuw i32 %index, 2
; FORCED-NEXT:    %6 = icmp eq i32 %index.next, 1000
; FORCED-NEXT:    br i1 %6, label %middle.block, label %vector.body

define void @test1(ptr %dst, {i64, i64} %sv) {
entry:
  br label %loop.body

loop.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.body ]
  %a = extractvalue { i64, i64 } %sv, 0
  %b = extractvalue { i64, i64 } %sv, 1
  %addr = getelementptr i64, ptr %dst, i32 %iv
  %add = add i64 %a, %b
  store i64 %add, ptr %addr
  %iv.next = add nsw i32 %iv, 1
  %cond = icmp ne i32 %iv.next, 1000
  br i1 %cond, label %loop.body, label %exit

exit:
  ret void
}


; Similar to the test case above, but checks getVectorCallCost as well.
declare float @powf(float, float) readnone nounwind

; CM: LV: Found uniform instruction:   %a = extractvalue { float, float } %sv, 0
; CM: LV: Found uniform instruction:   %b = extractvalue { float, float } %sv, 1

; CM: LV: Scalar loop costs: 14.
; CM: LV: Found an estimated cost of 0 for VF 2 For instruction:   %a = extractvalue { float, float } %sv, 0
; CM-NEXT: LV: Found an estimated cost of 0 for VF 2 For instruction:   %b = extractvalue { float, float } %sv, 1

; FORCED-LABEL: define void @test_getVectorCallCost

; FORCED-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; FORCED-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; FORCED-NEXT:    %0 = add i32 %index, 0
; FORCED-NEXT:    %1 = extractvalue { float, float } %sv, 0
; FORCED-NEXT:    %broadcast.splatinsert = insertelement <2 x float> poison, float %1, i64 0
; FORCED-NEXT:    %broadcast.splat = shufflevector <2 x float> %broadcast.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    %2 = extractvalue { float, float } %sv, 1
; FORCED-NEXT:    %broadcast.splatinsert1 = insertelement <2 x float> poison, float %2, i64 0
; FORCED-NEXT:    %broadcast.splat2 = shufflevector <2 x float> %broadcast.splatinsert1, <2 x float> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    %3 = getelementptr float, ptr %dst, i32 %0
; FORCED-NEXT:    %4 = call <2 x float> @llvm.pow.v2f32(<2 x float> %broadcast.splat, <2 x float> %broadcast.splat2)
; FORCED-NEXT:    %5 = getelementptr float, ptr %3, i32 0
; FORCED-NEXT:    store <2 x float> %4, ptr %5, align 4
; FORCED-NEXT:    %index.next = add nuw i32 %index, 2
; FORCED-NEXT:    %6 = icmp eq i32 %index.next, 1000
; FORCED-NEXT:    br i1 %6, label %middle.block, label %vector.body

define void @test_getVectorCallCost(ptr %dst, {float, float} %sv) {
entry:
  br label %loop.body

loop.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.body ]
  %a = extractvalue { float, float } %sv, 0
  %b = extractvalue { float, float } %sv, 1
  %addr = getelementptr float, ptr %dst, i32 %iv
  %p = call float @powf(float %a, float %b)
  store float %p, ptr %addr
  %iv.next = add nsw i32 %iv, 1
  %cond = icmp ne i32 %iv.next, 1000
  br i1 %cond, label %loop.body, label %exit

exit:
  ret void
}
