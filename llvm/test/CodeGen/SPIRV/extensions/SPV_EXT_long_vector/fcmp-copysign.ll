; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; fcmp and copysign on vectors whose length is not a power of two are widened to
; the next power of two.

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#V8F32:]] = OpTypeVector %[[#F32]] 8
; CHECK-DAG: %[[#V8Bool:]] = OpTypeVector %[[#Bool]] 8
; CHECK-DAG: %[[#V16F32:]] = OpTypeVector %[[#F32]] 16

; CHECK-LABEL: Begin function fcmp_v5f32
; CHECK: %[[#A:]] = OpCompositeConstruct %[[#V8F32]]
; CHECK: %[[#B:]] = OpCompositeConstruct %[[#V8F32]]
; CHECK: OpFOrdLessThan %[[#V8Bool]] %[[#A]] %[[#B]]
define <5 x i1> @fcmp_v5f32(<5 x float> %a, <5 x float> %b) {
  %r = fcmp olt <5 x float> %a, %b
  ret <5 x i1> %r
}

; CHECK-LABEL: Begin function copysign_v5f32
; CHECK: %[[#A:]] = OpCompositeConstruct %[[#V8F32]]
; CHECK: %[[#B:]] = OpCompositeConstruct %[[#V8F32]]
; CHECK: OpExtInst %[[#V8F32]] %[[#]] copysign %[[#A]] %[[#B]]
define <5 x float> @copysign_v5f32(<5 x float> %a, <5 x float> %b) {
  %r = call <5 x float> @llvm.copysign.v5f32(<5 x float> %a, <5 x float> %b)
  ret <5 x float> %r
}

; Longer than the maximum vector size: split into <16 x float> and a scalar.
; CHECK-LABEL: Begin function copysign_v17f32
; CHECK: OpExtInst %[[#V16F32]] %[[#]] copysign
; CHECK: OpExtInst %[[#F32]] %[[#]] copysign
; CHECK: OpReturnValue
define <17 x float> @copysign_v17f32(<17 x float> %a, <17 x float> %b) {
  %r = call <17 x float> @llvm.copysign.v17f32(<17 x float> %a, <17 x float> %b)
  ret <17 x float> %r
}
