; Test strict conversions between integer and float elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double>, metadata)
declare <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double>, metadata)
declare <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64>, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64>, metadata, metadata)

declare <2 x i32> @llvm.experimental.constrained.fptoui.v2i32.v2f64(<2 x double>, metadata)
declare <2 x i32> @llvm.experimental.constrained.fptosi.v2i32.v2f64(<2 x double>, metadata)
declare <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i32(<2 x i32>, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i32(<2 x i32>, metadata, metadata)

declare <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f32(<2 x float>, metadata)
declare <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f32(<2 x float>, metadata)
declare <2 x float> @llvm.experimental.constrained.uitofp.v2f32.v2i64(<2 x i64>, metadata, metadata)
declare <2 x float> @llvm.experimental.constrained.sitofp.v2f32.v2i64(<2 x i64>, metadata, metadata)

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

; Test conversion of signed i64s to f64s.
define <2 x double> @f3(<2 x i64> %dwords) #0 {
; CHECK-LABEL: f3:
; CHECK: vcdgb %v24, %v24, 0, 0
; CHECK: br %r14
  %doubles = call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64> %dwords,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret <2 x double> %doubles
}

; Test conversion of unsigned i64s to f64s.
define <2 x double> @f4(<2 x i64> %dwords) #0 {
; CHECK-LABEL: f4:
; CHECK: vcdlgb %v24, %v24, 0, 0
; CHECK: br %r14
  %doubles = call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64> %dwords,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret <2 x double> %doubles
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

; Test conversion of signed i32s to f64s, which must compile.
define <2 x double> @f7(<2 x i32> *%ptr) #0 {
  %words = load <2 x i32>, <2 x i32> *%ptr
  %doubles = call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i32(<2 x i32> %words,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret <2 x double> %doubles
}

; Test conversion of unsigned i32s to f64s, which must compile.
define <2 x double> @f8(<2 x i32> *%ptr) #0 {
  %words = load <2 x i32>, <2 x i32> *%ptr
  %doubles = call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i32(<2 x i32> %words,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret <2 x double> %doubles
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

; Test conversion of signed i64s to f32, which must compile.
define void @f11(<2 x i64> %dwords, <2 x float> *%ptr) #0 {
  %floats = call <2 x float> @llvm.experimental.constrained.sitofp.v2f32.v2i64(<2 x i64> %dwords,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store <2 x float> %floats, <2 x float> *%ptr
  ret void
}

; Test conversion of unsigned i64s to f32, which must compile.
define void @f12(<2 x i64> %dwords, <2 x float> *%ptr) #0 {
  %floats = call <2 x float> @llvm.experimental.constrained.uitofp.v2f32.v2i64(<2 x i64> %dwords,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store <2 x float> %floats, <2 x float> *%ptr
  ret void
}

attributes #0 = { strictfp }
