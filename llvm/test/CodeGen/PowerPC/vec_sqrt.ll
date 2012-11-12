; RUN: llc -mcpu=pwr6 -mattr=+altivec,+fsqrt < %s | FileCheck %s

; Check for vector sqrt expansion using floating-point types, since altivec
; does not provide an fsqrt instruction for vector.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <2 x float> @llvm.sqrt.v2f32(<2 x float> %val)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float> %val)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float> %val)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double> %val)
declare <4 x double> @llvm.sqrt.v4f64(<4 x double> %val)

define <2 x float> @v2f32_sqrt(<2 x float> %x) nounwind readnone {
entry:
  %sqrt = call <2 x float> @llvm.sqrt.v2f32 (<2 x float> %x)
  ret <2 x float> %sqrt
}
; sqrt (<2 x float>) is promoted to sqrt (<4 x float>)
; CHECK: v2f32_sqrt:
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}

define <4 x float> @v4f32_sqrt(<4 x float> %x) nounwind readnone {
entry:
  %sqrt = call <4 x float> @llvm.sqrt.v4f32 (<4 x float> %x)
  ret <4 x float> %sqrt
}
; CHECK: v4f32_sqrt:
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}

define <8 x float> @v8f32_sqrt(<8 x float> %x) nounwind readnone {
entry:
  %sqrt = call <8 x float> @llvm.sqrt.v8f32 (<8 x float> %x)
  ret <8 x float> %sqrt
}
; CHECK: v8f32_sqrt:
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrts {{[0-9]+}}, {{[0-9]+}}

define <2 x double> @v2f64_sqrt(<2 x double> %x) nounwind readnone {
entry:
  %sqrt = call <2 x double> @llvm.sqrt.v2f64 (<2 x double> %x)
  ret <2 x double> %sqrt
}
; CHECK: v2f64_sqrt:
; CHECK: fsqrt {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrt {{[0-9]+}}, {{[0-9]+}}

define <4 x double> @v4f64_sqrt(<4 x double> %x) nounwind readnone {
entry:
  %sqrt = call <4 x double> @llvm.sqrt.v4f64 (<4 x double> %x)
  ret <4 x double> %sqrt
}
; CHECK: v4f64_sqrt:
; CHECK: fsqrt {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrt {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrt {{[0-9]+}}, {{[0-9]+}}
; CHECK: fsqrt {{[0-9]+}}, {{[0-9]+}}
