; RUN: llc -verify-machineinstrs -mcpu=pwr6 -mattr=+altivec < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <2 x float> @llvm.fmuladd.v2f32(<2 x float> %val, <2 x float>, <2 x float>)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float> %val, <4 x float>, <4 x float>)
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float> %val, <8 x float>, <8 x float>)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double> %val, <2 x double>, <2 x double>)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double> %val, <4 x double>, <4 x double>)

define <2 x float> @v2f32_fmuladd(<2 x float> %x) nounwind readnone {
entry:
  %fmuladd = call <2 x float> @llvm.fmuladd.v2f32 (<2 x float> %x, <2 x float> %x, <2 x float> %x)
  ret <2 x float> %fmuladd
}
; fmuladd (<2 x float>) is promoted to fmuladd (<4 x float>)
; CHECK-LABEL: v2f32_fmuladd:
; CHECK: vmaddfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}

define <4 x float> @v4f32_fmuladd(<4 x float> %x) nounwind readnone {
entry:
  %fmuladd = call <4 x float> @llvm.fmuladd.v4f32 (<4 x float> %x, <4 x float> %x, <4 x float> %x)
  ret <4 x float> %fmuladd
}
; CHECK-LABEL: v4f32_fmuladd:
; CHECK: vmaddfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}

define <8 x float> @v8f32_fmuladd(<8 x float> %x) nounwind readnone {
entry:
  %fmuladd = call <8 x float> @llvm.fmuladd.v8f32 (<8 x float> %x, <8 x float> %x, <8 x float> %x)
  ret <8 x float> %fmuladd
}
; CHECK-LABEL: v8f32_fmuladd:
; CHECK: vmaddfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vmaddfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}

define <2 x double> @v2f64_fmuladd(<2 x double> %x) nounwind readnone {
entry:
  %fmuladd = call <2 x double> @llvm.fmuladd.v2f64 (<2 x double> %x, <2 x double> %x, <2 x double> %x)
  ret <2 x double> %fmuladd
}
; CHECK-LABEL: v2f64_fmuladd:
; CHECK: fmadd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: fmadd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}

define <4 x double> @v4f64_fmuladd(<4 x double> %x) nounwind readnone {
entry:
  %fmuladd = call <4 x double> @llvm.fmuladd.v4f64 (<4 x double> %x, <4 x double> %x, <4 x double> %x)
  ret <4 x double> %fmuladd
}
; CHECK-LABEL: v4f64_fmuladd:
; CHECK: fmadd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}} 
; CHECK: fmadd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}} 
; CHECK: fmadd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}} 
; CHECK: fmadd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}} 
