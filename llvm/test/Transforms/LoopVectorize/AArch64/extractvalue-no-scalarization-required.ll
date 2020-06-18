; REQUIRES: asserts

; RUN: opt -loop-vectorize -mtriple=arm64-apple-ios %s -S -debug -disable-output 2>&1 | FileCheck --check-prefix=CM %s
; RUN: opt -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 %s -S | FileCheck --check-prefix=FORCED %s

; Test case from PR41294.

; Check scalar cost for extractvalue. The constant and loop invariant operands are free,
; leaving cost 3 for scalarizing the result + 2 for executing the op with VF 2.

; CM: LV: Scalar loop costs: 9.
; CM: LV: Found an estimated cost of 5 for VF 2 For instruction:   %a = extractvalue { i64, i64 } %sv, 0
; CM-NEXT: LV: Found an estimated cost of 5 for VF 2 For instruction:   %b = extractvalue { i64, i64 } %sv, 1

; Check that the extractvalue operands are actually free in vector code.

; FORCED-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; FORCED-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; FORCED-NEXT:    %0 = add i32 %index, 0
; FORCED-NEXT:    %1 = extractvalue { i64, i64 } %sv, 0
; FORCED-NEXT:    %2 = extractvalue { i64, i64 } %sv, 0
; FORCED-NEXT:    %3 = insertelement <2 x i64> undef, i64 %1, i32 0
; FORCED-NEXT:    %4 = insertelement <2 x i64> %3, i64 %2, i32 1
; FORCED-NEXT:    %5 = extractvalue { i64, i64 } %sv, 1
; FORCED-NEXT:    %6 = extractvalue { i64, i64 } %sv, 1
; FORCED-NEXT:    %7 = insertelement <2 x i64> undef, i64 %5, i32 0
; FORCED-NEXT:    %8 = insertelement <2 x i64> %7, i64 %6, i32 1
; FORCED-NEXT:    %9 = getelementptr i64, i64* %dst, i32 %0
; FORCED-NEXT:    %10 = add <2 x i64> %4, %8
; FORCED-NEXT:    %11 = getelementptr i64, i64* %9, i32 0
; FORCED-NEXT:    %12 = bitcast i64* %11 to <2 x i64>*
; FORCED-NEXT:    store <2 x i64> %10, <2 x i64>* %12, align 4
; FORCED-NEXT:    %index.next = add i32 %index, 2
; FORCED-NEXT:    %13 = icmp eq i32 %index.next, 0
; FORCED-NEXT:    br i1 %13, label %middle.block, label %vector.body, !llvm.loop !0

define void @test1(i64* %dst, {i64, i64} %sv) {
entry:
  br label %loop.body

loop.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.body ]
  %a = extractvalue { i64, i64 } %sv, 0
  %b = extractvalue { i64, i64 } %sv, 1
  %addr = getelementptr i64, i64* %dst, i32 %iv
  %add = add i64 %a, %b
  store i64 %add, i64* %addr
  %iv.next = add nsw i32 %iv, 1
  %cond = icmp ne i32 %iv.next, 0
  br i1 %cond, label %loop.body, label %exit

exit:
  ret void
}


; Similar to the test case above, but checks getVectorCallCost as well.
declare float @pow(float, float) readnone nounwind

; CM: LV: Scalar loop costs: 18.
; CM: LV: Found an estimated cost of 5 for VF 2 For instruction:   %a = extractvalue { float, float } %sv, 0
; CM-NEXT: LV: Found an estimated cost of 5 for VF 2 For instruction:   %b = extractvalue { float, float } %sv, 1

; FORCED-LABEL: define void @test_getVectorCallCost

; FORCED-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; FORCED-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; FORCED-NEXT:    %0 = add i32 %index, 0
; FORCED-NEXT:    %1 = extractvalue { float, float } %sv, 0
; FORCED-NEXT:    %2 = extractvalue { float, float } %sv, 0
; FORCED-NEXT:    %3 = insertelement <2 x float> undef, float %1, i32 0
; FORCED-NEXT:    %4 = insertelement <2 x float> %3, float %2, i32 1
; FORCED-NEXT:    %5 = extractvalue { float, float } %sv, 1
; FORCED-NEXT:    %6 = extractvalue { float, float } %sv, 1
; FORCED-NEXT:    %7 = insertelement <2 x float> undef, float %5, i32 0
; FORCED-NEXT:    %8 = insertelement <2 x float> %7, float %6, i32 1
; FORCED-NEXT:    %9 = getelementptr float, float* %dst, i32 %0
; FORCED-NEXT:    %10 = call <2 x float> @llvm.pow.v2f32(<2 x float> %4, <2 x float> %8)
; FORCED-NEXT:    %11 = getelementptr float, float* %9, i32 0
; FORCED-NEXT:    %12 = bitcast float* %11 to <2 x float>*
; FORCED-NEXT:    store <2 x float> %10, <2 x float>* %12, align 4
; FORCED-NEXT:    %index.next = add i32 %index, 2
; FORCED-NEXT:    %13 = icmp eq i32 %index.next, 0
; FORCED-NEXT:    br i1 %13, label %middle.block, label %vector.body, !llvm.loop !4

define void @test_getVectorCallCost(float* %dst, {float, float} %sv) {
entry:
  br label %loop.body

loop.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.body ]
  %a = extractvalue { float, float } %sv, 0
  %b = extractvalue { float, float } %sv, 1
  %addr = getelementptr float, float* %dst, i32 %iv
  %p = call float @pow(float %a, float %b)
  store float %p, float* %addr
  %iv.next = add nsw i32 %iv, 1
  %cond = icmp ne i32 %iv.next, 0
  br i1 %cond, label %loop.body, label %exit

exit:
  ret void
}



