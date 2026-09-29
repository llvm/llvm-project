; Test strict conversions between integer and float elements on z15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f32(<4 x float>, metadata)
declare <4 x i32> @llvm.experimental.constrained.fptosi.v4i32.v4f32(<4 x float>, metadata)
declare <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i32(<4 x i32>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.sitofp.v4f32.v4i32(<4 x i32>, metadata, metadata)

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

; Test conversion of signed i32s to f32s.
define <4 x float> @f3(<4 x i32> %dwords) #0 {
; CHECK-LABEL: f3:
; CHECK: vcefb %v24, %v24, 0, 0
; CHECK: br %r14
  %floats = call <4 x float> @llvm.experimental.constrained.sitofp.v4f32.v4i32(<4 x i32> %dwords,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret <4 x float> %floats
}

; Test conversion of unsigned i32s to f32s.
define <4 x float> @f4(<4 x i32> %dwords) #0 {
; CHECK-LABEL: f4:
; CHECK: vcelfb %v24, %v24, 0, 0
; CHECK: br %r14
  %floats = call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i32(<4 x i32> %dwords,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret <4 x float> %floats
}

attributes #0 = { strictfp }
