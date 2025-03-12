; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-ios %s -S -debug -disable-output 2>&1 | FileCheck --check-prefix=CM %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 %s -S | FileCheck --check-prefix=FORCED %s

; Test case from PR41294.

; Check scalar cost for extractvalue. The constant and loop invariant operands are free,
; leaving cost 3 for scalarizing the result + 2 for executing the op with VF 2.

; CM: LV: Found uniform instruction:   %a = extractvalue { i64, i64 } %sv, 0
; CM: LV: Found uniform instruction:   %b = extractvalue { i64, i64 } %sv, 1

; Ensure the extractvalue + add instructions are hoisted out
; CM: vector.ph:
; CM:  CLONE ir<%a> = extractvalue ir<%sv>
; CM:  CLONE ir<%b> = extractvalue ir<%sv>
; CM:  WIDEN ir<%add> = add ir<%a>, ir<%b>
; CM:  Successor(s): vector loop

; CM: LV: Scalar loop costs: 5.

; Check that the extractvalue operands are actually free in vector code.

; FORCED:         [[E1:%.+]] = extractvalue { i64, i64 } %sv, 0
; FORCED-NEXT:    [[E2:%.+]] = extractvalue { i64, i64 } %sv, 1
; FORCED-NEXT:    %broadcast.splatinsert = insertelement <2 x i64> poison, i64 [[E1]], i64 0
; FORCED-NEXT:    %broadcast.splat = shufflevector <2 x i64> %broadcast.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    %broadcast.splatinsert1 = insertelement <2 x i64> poison, i64 [[E2]], i64 0
; FORCED-NEXT:    %broadcast.splat2 = shufflevector <2 x i64> %broadcast.splatinsert1, <2 x i64> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    [[ADD:%.+]] = add <2 x i64> %broadcast.splat, %broadcast.splat2

; FORCED-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; FORCED-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; FORCED-NEXT:    [[IV_0:%.]] = add i32 %index, 0
; FORCED-NEXT:    [[GEP:%.+]] = getelementptr i64, ptr %dst, i32 [[IV_0]]
; FORCED-NEXT:    [[GEP2:%.+]] = getelementptr i64, ptr [[GEP]], i32 0
; FORCED-NEXT:    store <2 x i64> [[ADD]], ptr [[GEP2]], align 4
; FORCED-NEXT:    %index.next = add nuw i32 %index, 2
; FORCED-NEXT:    [[C:%.+]] = icmp eq i32 %index.next, 1000
; FORCED-NEXT:    br i1 [[C]], label %middle.block, label %vector.body

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

; Ensure the extractvalue instructions are hoisted out
; CM-LABEL: Checking a loop in 'test_getVectorCallCost'
; CM: vector.ph:
; CM:  CLONE ir<%a> = extractvalue ir<%sv>
; CM:  CLONE ir<%b> = extractvalue ir<%sv>
; CM:  Successor(s): vector loop

; CM: LV: Scalar loop costs: 14.

; FORCED-LABEL: define void @test_getVectorCallCost

; FORCED:         [[E1:%.+]] = extractvalue { float, float } %sv, 0
; FORCED-NEXT:    [[E2:%.+]] = extractvalue { float, float } %sv, 1
; FORCED-NEXT:    %broadcast.splatinsert = insertelement <2 x float> poison, float [[E1]], i64 0
; FORCED-NEXT:    %broadcast.splat = shufflevector <2 x float> %broadcast.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
; FORCED-NEXT:    %broadcast.splatinsert1 = insertelement <2 x float> poison, float [[E2]], i64 0
; FORCED-NEXT:    %broadcast.splat2 = shufflevector <2 x float> %broadcast.splatinsert1, <2 x float> poison, <2 x i32> zeroinitializer

; FORCED-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; FORCED-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; FORCED-NEXT:    [[IV0:%.+]] = add i32 %index, 0
; FORCED-NEXT:    [[GEP1:%.+]] = getelementptr float, ptr %dst, i32 [[IV0]]
; FORCED-NEXT:    [[POW:%.+]] = call <2 x float> @llvm.pow.v2f32(<2 x float> %broadcast.splat, <2 x float> %broadcast.splat2)
; FORCED-NEXT:    [[GEP2:%.+]] = getelementptr float, ptr [[GEP1]], i32 0
; FORCED-NEXT:    store <2 x float> [[POW]], ptr [[GEP2]], align 4
; FORCED-NEXT:    %index.next = add nuw i32 %index, 2
; FORCED-NEXT:    [[C:%.+]] = icmp eq i32 %index.next, 1000
; FORCED-NEXT:    br i1 [[C]], label %middle.block, label %vector.body

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
