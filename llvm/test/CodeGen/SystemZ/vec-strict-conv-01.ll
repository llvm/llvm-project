; Test strict conversions between integer and float elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; FIXME: llvm.experimental.constrained.[su]itofp does not yet exist

declare <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double>, metadata)
declare <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double>, metadata)

declare <2 x i32> @llvm.experimental.constrained.fptoui.v2i32.v2f64(<2 x double>, metadata)
declare <2 x i32> @llvm.experimental.constrained.fptosi.v2i32.v2f64(<2 x double>, metadata)

declare <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f32(<2 x float>, metadata)
declare <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f32(<2 x float>, metadata)

; Test conversion of f64s to signed i64s.
define <2 x i64> @f1(<2 x double> %doubles) #0 {
; CHECK-LABEL: f1:
; CHECK: vcgdb %v24, %v24, 0, 5
; CHECK: br %r14
  %dwords = call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double> %doubles,
                                               metadata !"fpexcept.strict") #0
  ret <2 x i64> %dwords
}

; Test conversion of f64s to unsigned i64s.
define <2 x i64> @f2(<2 x double> %doubles) #0 {
; CHECK-LABEL: f2:
; CHECK: vclgdb %v24, %v24, 0, 5
; CHECK: br %r14
  %dwords = call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double> %doubles,
                                               metadata !"fpexcept.strict") #0
  ret <2 x i64> %dwords
}

; Test conversion of f64s to signed i32s, which must compile.
define void @f5(<2 x double> %doubles, <2 x i32> *%ptr) #0 {
  %words = call <2 x i32> @llvm.experimental.constrained.fptosi.v2i32.v2f64(<2 x double> %doubles,
                                               metadata !"fpexcept.strict") #0
  store <2 x i32> %words, <2 x i32> *%ptr
  ret void
}

; Test conversion of f64s to unsigned i32s, which must compile.
define void @f6(<2 x double> %doubles, <2 x i32> *%ptr) #0 {
  %words = call <2 x i32> @llvm.experimental.constrained.fptoui.v2i32.v2f64(<2 x double> %doubles,
                                               metadata !"fpexcept.strict") #0
  store <2 x i32> %words, <2 x i32> *%ptr
  ret void
}

; Test conversion of f32s to signed i64s, which must compile.
define <2 x i64> @f9(<2 x float> *%ptr) #0 {
  %floats = load <2 x float>, <2 x float> *%ptr
  %dwords = call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f32(<2 x float> %floats,
                                               metadata !"fpexcept.strict") #0
  ret <2 x i64> %dwords
}

; Test conversion of f32s to unsigned i64s, which must compile.
define <2 x i64> @f10(<2 x float> *%ptr) #0 {
  %floats = load <2 x float>, <2 x float> *%ptr
  %dwords = call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f32(<2 x float> %floats,
                                               metadata !"fpexcept.strict") #0
  ret <2 x i64> %dwords
}

attributes #0 = { strictfp }
