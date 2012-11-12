; RUN: opt < %s -scalarrepl -S | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: @test1
; CHECK: %[[alloc:[\.a-z0-9]*]] = alloca <4 x float>
; CHECK: store <4 x float> zeroinitializer, <4 x float>* %[[alloc]]
; CHECK: memset
; CHECK: extractelement <4 x float> zeroinitializer, i32 %idx2

; Split the array but don't replace the memset with an insert
; element as its not a constant offset.
; The load, however, can be replaced with an extract element.
define float @test1(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca [4 x <4 x float>]
  store [4 x <4 x float>] zeroinitializer, [4 x <4 x float>]* %0
  %ptr1 = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0, i32 %idx1
  %cast = bitcast float* %ptr1 to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 4, i32 4, i1 false)
  %ptr2 = getelementptr [4 x <4 x float>]* %0, i32 0, i32 1, i32 %idx2
  %ret = load float* %ptr2
  ret float %ret
}

; CHECK: @test2
; CHECK: %[[ins:[\.a-z0-9]*]] = insertelement <4 x float> zeroinitializer, float 1.000000e+00, i32 %idx1
; CHECK: extractelement <4 x float> %[[ins]], i32 %idx2

; Do SROA on the array when it has dynamic vector reads and writes.
define float @test2(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca [4 x <4 x float>]
  store [4 x <4 x float>] zeroinitializer, [4 x <4 x float>]* %0
  %ptr1 = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0, i32 %idx1
  store float 1.0, float* %ptr1
  %ptr2 = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0, i32 %idx2
  %ret = load float* %ptr2
  ret float %ret
}

; CHECK: test3
; CHECK: %0 = alloca [4 x <4 x float>]
; CHECK-NOT: alloca

; Don't do SROA on a dynamically indexed vector when it spans
; more than one array element of the alloca array it is within.
define float @test3(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca [4 x <4 x float>]
  store [4 x <4 x float>] zeroinitializer, [4 x <4 x float>]* %0
  %bigvec = bitcast [4 x <4 x float>]* %0 to <16 x float>*
  %ptr1 = getelementptr <16 x float>* %bigvec, i32 0, i32 %idx1
  store float 1.0, float* %ptr1
  %ptr2 = getelementptr <16 x float>* %bigvec, i32 0, i32 %idx2
  %ret = load float* %ptr2
  ret float %ret
}

; CHECK: test4
; CHECK: insertelement <16 x float> zeroinitializer, float 1.000000e+00, i32 %idx1
; CHECK: extractelement <16 x float> %0, i32 %idx2

; Don't do SROA on a dynamically indexed vector when it spans
; more than one array element of the alloca array it is within.
; However, unlike test3, the store is on the vector type
; so SROA will convert the large alloca into the large vector
; type and do all accesses with insert/extract element
define float @test4(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca [4 x <4 x float>]
  %bigvec = bitcast [4 x <4 x float>]* %0 to <16 x float>*
  store <16 x float> zeroinitializer, <16 x float>* %bigvec
  %ptr1 = getelementptr <16 x float>* %bigvec, i32 0, i32 %idx1
  store float 1.0, float* %ptr1
  %ptr2 = getelementptr <16 x float>* %bigvec, i32 0, i32 %idx2
  %ret = load float* %ptr2
  ret float %ret
}

; CHECK: @test5
; CHECK: %0 = alloca [4 x <4 x float>]
; CHECK-NOT: alloca

; Don't do SROA as the is a second dynamically indexed array
; which may span multiple elements of the alloca.
define float @test5(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca [4 x <4 x float>]
  store [4 x <4 x float>] zeroinitializer, [4 x <4 x float>]* %0
  %ptr1 = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0, i32 %idx1
  %ptr2 = bitcast float* %ptr1 to [1 x <2 x float>]*
  %ptr3 = getelementptr [1 x <2 x float>]* %ptr2, i32 0, i32 0, i32 %idx1
  store float 1.0, float* %ptr1
  %ptr4 = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0, i32 %idx2
  %ret = load float* %ptr4
  ret float %ret
}

; CHECK: test6
; CHECK: insertelement <4 x float> zeroinitializer, float 1.000000e+00, i32 %idx1
; CHECK: extractelement <4 x float> zeroinitializer, i32 %idx2

%vector.pair = type { %vector.anon, %vector.anon }
%vector.anon = type { %vector }
%vector = type { <4 x float> }

; Dynamic GEPs on vectors were crashing when the vector was inside a struct
; as the new GEP for the new alloca might not include all the indices from
; the original GEP, just the indices it needs to get to the correct offset of
; some type, not necessarily the dynamic vector.
; This test makes sure we don't have this crash.
define float @test6(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca %vector.pair
  store %vector.pair zeroinitializer, %vector.pair* %0
  %ptr1 = getelementptr %vector.pair* %0, i32 0, i32 0, i32 0, i32 0, i32 %idx1
  store float 1.0, float* %ptr1
  %ptr2 = getelementptr %vector.pair* %0, i32 0, i32 1, i32 0, i32 0, i32 %idx2
  %ret = load float* %ptr2
  ret float %ret
}

; CHECK: test7
; CHECK: insertelement <4 x float> zeroinitializer, float 1.000000e+00, i32 %idx1
; CHECK: extractelement <4 x float> zeroinitializer, i32 %idx2

%array.pair = type { [2 x %array.anon], %array.anon }
%array.anon = type { [2 x %vector] }

; This is the same as test6 and tests the same crash, but on arrays.
define float @test7(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca %array.pair
  store %array.pair zeroinitializer, %array.pair* %0
  %ptr1 = getelementptr %array.pair* %0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %idx1
  store float 1.0, float* %ptr1
  %ptr2 = getelementptr %array.pair* %0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 %idx2
  %ret = load float* %ptr2
  ret float %ret
}

; CHECK: test8
; CHECK: %[[offset1:[\.a-z0-9]*]] = add i32 %idx1, 1
; CHECK: %[[ins:[\.a-z0-9]*]] = insertelement <4 x float> zeroinitializer, float 1.000000e+00, i32 %[[offset1]]
; CHECK: %[[offset2:[\.a-z0-9]*]] = add i32 %idx2, 2
; CHECK: extractelement <4 x float> %[[ins]], i32 %[[offset2]]

; Do SROA on the vector when it has dynamic vector reads and writes
; from a non-zero offset.
define float @test8(i32 %idx1, i32 %idx2) {
entry:
  %0 = alloca <4 x float>
  store <4 x float> zeroinitializer, <4 x float>* %0
  %ptr1 = getelementptr <4 x float>* %0, i32 0, i32 1
  %ptr2 = bitcast float* %ptr1 to <3 x float>*
  %ptr3 = getelementptr <3 x float>* %ptr2, i32 0, i32 %idx1
  store float 1.0, float* %ptr3
  %ptr4 = getelementptr <4 x float>* %0, i32 0, i32 2
  %ptr5 = bitcast float* %ptr4 to <2 x float>*
  %ptr6 = getelementptr <2 x float>* %ptr5, i32 0, i32 %idx2
  %ret = load float* %ptr6
  ret float %ret
}

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i32, i1)
