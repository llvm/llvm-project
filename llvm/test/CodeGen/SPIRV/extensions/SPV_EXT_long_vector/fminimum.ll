; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; llvm.minimum on <5 x float> is widened to <8 x float> before being lowered, so
; the signed-zero check operates on newly created, widened operands.

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V8F32:]] = OpTypeVector %[[#F32]] 8
; CHECK-DAG: %[[#V8I32:]] = OpTypeVector %[[#I32]] 8

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpCompositeConstruct %[[#V8F32]]
; CHECK: %[[#B:]] = OpCompositeConstruct %[[#V8F32]]
; CHECK: %[[#Min:]] = OpExtInst %[[#V8F32]] %[[#]] fmin %[[#A]] %[[#B]]
; CHECK: %[[#Ord:]] = OpOrdered %[[#]] %[[#A]] %[[#B]]
; CHECK: %[[#NaNSel:]] = OpSelect %[[#V8F32]] %[[#Ord]] %[[#Min]] %[[#]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#]] %[[#NaNSel]] %[[#]]
; CHECK: %[[#]] = OpBitcast %[[#V8I32]] %[[#A]]
; CHECK: %[[#]] = OpBitcast %[[#V8I32]] %[[#B]]
; CHECK: %[[#Res:]] = OpSelect %[[#V8F32]] %[[#IsZero]] %[[#]] %[[#NaNSel]]
; CHECK: OpCompositeExtract %[[#F32]] %[[#Res]] 0
; CHECK: OpReturnValue
define <5 x float> @minimum_v5f32(<5 x float> %a, <5 x float> %b) {
  %r = call <5 x float> @llvm.minimum.v5f32(<5 x float> %a, <5 x float> %b)
  ret <5 x float> %r
}
