; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The instructions that llvm.minimum is expanded into are widened to
; <8 x float> one by one.

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V8F32:]] = OpTypeVector %[[#F32]] 8

; CHECK-LABEL: Begin function minimum_v5f32
; CHECK: OpExtInst %[[#V8F32]] %[[#]] fmin
; CHECK: OpExtInst %[[#V8F32]] %[[#]] copysign
; CHECK: OpFOrdLessThan
; CHECK: OpSelect
; CHECK: OpExtInst %[[#V8F32]] %[[#]] copysign
; CHECK: OpUnordered
; CHECK: %[[#Res:]] = OpSelect
; CHECK: OpReturnValue %[[#Res]]
define <5 x float> @minimum_v5f32(<5 x float> %a, <5 x float> %b) {
  %r = call <5 x float> @llvm.minimum.v5f32(<5 x float> %a, <5 x float> %b)
  ret <5 x float> %r
}
