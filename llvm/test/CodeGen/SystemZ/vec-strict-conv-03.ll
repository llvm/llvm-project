; Test strict conversions between integer and float elements on z15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

; FIXME: llvm.experimental.constrained.[su]itofp does not yet exist

declare <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f32(<4 x float>, metadata)
declare <4 x i32> @llvm.experimental.constrained.fptosi.v4i32.v4f32(<4 x float>, metadata)

; Test conversion of f32s to signed i32s.
define <4 x i32> @f1(<4 x float> %floats) #0 {
; CHECK-LABEL: f1:
; CHECK: vcfeb %v24, %v24, 0, 5
; CHECK: br %r14
  %words = call <4 x i32> @llvm.experimental.constrained.fptosi.v4i32.v4f32(<4 x float> %floats,
                                               metadata !"fpexcept.strict") #0
  ret <4 x i32> %words
}

; Test conversion of f32s to unsigned i32s.
define <4 x i32> @f2(<4 x float> %floats) #0 {
; CHECK-LABEL: f2:
; CHECK: vclfeb %v24, %v24, 0, 5
; CHECK: br %r14
  %words = call <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f32(<4 x float> %floats,
                                               metadata !"fpexcept.strict") #0
  ret <4 x i32> %words
}

attributes #0 = { strictfp }
