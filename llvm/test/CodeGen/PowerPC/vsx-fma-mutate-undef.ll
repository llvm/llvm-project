; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind
define void @acosh_float8(<4 x i32> %v1, <4 x i32> %v2) #0 {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> undef, <4 x float> <float 0x3FE62E4200000000, float 0x3FE62E4200000000, float 0x3FE62E4200000000, float 0x3FE62E4200000000>, <4 x float> undef) #0
  %astype.i.i.74.i = bitcast <4 x float> %0 to <4 x i32>
  %and.i.i.76.i = and <4 x i32> %astype.i.i.74.i, %v1
  %or.i.i.79.i = or <4 x i32> %and.i.i.76.i, %v2
  %astype5.i.i.80.i = bitcast <4 x i32> %or.i.i.79.i to <4 x float>
  %1 = shufflevector <4 x float> %astype5.i.i.80.i, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <8 x float> undef, <8 x float> %1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  store <8 x float> %2, <8 x float>* undef, align 32
  br label %if.end

; CHECK-LABEL: @acosh_float8
; CHECK: xvmaddasp

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

