; RUN: opt < %s -passes=slp-vectorizer -o - -S -slp-threshold=-1000 | FileCheck %s

target datalayout = "e-p:32:32-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx--nvidiacl"

; Test that CTLZ can be vectorized currently even though the second argument is a scalar

define <2 x i8> @cltz_test(<2 x i8> %x) #0 {
; CHECK-LABEL: @cltz_test(
;      CHECK: [[VEC:%.*]] = call <2 x i8> @llvm.ctlz.v2i8(<2 x i8> %{{.*}}, i1 false)
; CHECK-NEXT: ret <2 x i8> [[VEC]]
;
entry:
  %0 = extractelement <2 x i8> %x, i32 0
  %call.i = call i8 @llvm.ctlz.i8(i8 %0, i1 false)
  %vecinit = insertelement <2 x i8> undef, i8 %call.i, i32 0
  %1 = extractelement <2 x i8> %x, i32 1
  %call.i4 = call i8 @llvm.ctlz.i8(i8 %1, i1 false)
  %vecinit2 = insertelement <2 x i8> %vecinit, i8 %call.i4, i32 1
  ret <2 x i8> %vecinit2
}


define <2 x i8> @cltz_test_poison(<2 x i8> %x) #0 {
; CHECK-LABEL: @cltz_test_poison(
;      CHECK: [[VEC:%.*]] = call <2 x i8> @llvm.ctlz.v2i8(<2 x i8> %{{.*}}, i1 false)
; CHECK-NEXT: ret <2 x i8> [[VEC]]
;
entry:
  %0 = extractelement <2 x i8> %x, i32 0
  %call.i = call i8 @llvm.ctlz.i8(i8 %0, i1 false)
  %vecinit = insertelement <2 x i8> poison, i8 %call.i, i32 0
  %1 = extractelement <2 x i8> %x, i32 1
  %call.i4 = call i8 @llvm.ctlz.i8(i8 %1, i1 false)
  %vecinit2 = insertelement <2 x i8> %vecinit, i8 %call.i4, i32 1
  ret <2 x i8> %vecinit2
}

declare i8 @llvm.ctlz.i8(i8, i1) #3

attributes #0 = { alwaysinline nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
