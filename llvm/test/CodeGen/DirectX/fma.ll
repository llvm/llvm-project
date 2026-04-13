; RUN: opt -S -scalarizer -dxil-op-lower < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64"
target triple = "dxil-pc-shadermodel6.7-library"

; CHECK-LABEL: define double @fma_double(
; CHECK: call double @dx.op.tertiary.f64(i32 47, double %{{.*}}, double %{{.*}}, double %{{.*}}) #[[#ATTR:]]
define double @fma_double(double %a, double %b, double %c) {
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

; CHECK-LABEL: define <2 x double> @fma_v2f64(
; CHECK: extractelement <2 x double> %a, i64 0
; CHECK: extractelement <2 x double> %b, i64 0
; CHECK: extractelement <2 x double> %c, i64 0
; CHECK: call double @dx.op.tertiary.f64(i32 47, double %{{.*}}, double %{{.*}}, double %{{.*}}) #[[#ATTR]]
; CHECK: extractelement <2 x double> %a, i64 1
; CHECK: extractelement <2 x double> %b, i64 1
; CHECK: extractelement <2 x double> %c, i64 1
; CHECK: call double @dx.op.tertiary.f64(i32 47, double %{{.*}}, double %{{.*}}, double %{{.*}}) #[[#ATTR]]
; CHECK: insertelement <2 x double> poison, double %{{.*}}, i64 0
; CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 1
define <2 x double> @fma_v2f64(<2 x double> %a, <2 x double> %b,
                                <2 x double> %c) {
  %r = call <2 x double> @llvm.fma.v2f64(<2 x double> %a, <2 x double> %b,
                                         <2 x double> %c)
  ret <2 x double> %r
}

; CHECK-LABEL: define <16 x double> @fma_v16f64(
; CHECK: extractelement <16 x double> %a, i64 0
; CHECK: extractelement <16 x double> %b, i64 0
; CHECK: extractelement <16 x double> %c, i64 0
; CHECK: call double @dx.op.tertiary.f64(i32 47, double %{{.*}}, double %{{.*}}, double %{{.*}}) #[[#ATTR]]
; CHECK: extractelement <16 x double> %a, i64 15
; CHECK: extractelement <16 x double> %b, i64 15
; CHECK: extractelement <16 x double> %c, i64 15
; CHECK: call double @dx.op.tertiary.f64(i32 47, double %{{.*}}, double %{{.*}}, double %{{.*}}) #[[#ATTR]]
; CHECK: insertelement <16 x double> poison, double %{{.*}}, i64 0
; CHECK: insertelement <16 x double> %{{.*}}, double %{{.*}}, i64 15
define <16 x double> @fma_v16f64(<16 x double> %a, <16 x double> %b,
                                  <16 x double> %c) {
  %r = call <16 x double> @llvm.fma.v16f64(<16 x double> %a, <16 x double> %b,
                                           <16 x double> %c)
  ret <16 x double> %r
}

declare double @llvm.fma.f64(double, double, double)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <16 x double> @llvm.fma.v16f64(<16 x double>, <16 x double>, <16 x double>)

; CHECK: attributes #[[#ATTR]] = { memory(none) }
