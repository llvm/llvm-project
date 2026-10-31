; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s

; The expansion of llvm.minimum/llvm.maximum must use bfloat types and
; constants.

; CHECK-DAG: %[[#Ext:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#BF16:]] = OpTypeFloat 16 0
; CHECK-DAG: %[[#V4BF16:]] = OpTypeVector %[[#BF16]] 4
; CHECK-DAG: %[[#One:]] = OpConstant %[[#BF16]] 16256
; CHECK-DAG: %[[#NaN:]] = OpConstant %[[#BF16]] 32704
; CHECK-DAG: %[[#V4One:]] = OpConstantComposite %[[#V4BF16]] %[[#One]] %[[#One]] %[[#One]] %[[#One]]
; CHECK-DAG: %[[#V4NaN:]] = OpConstantComposite %[[#V4BF16]] %[[#NaN]] %[[#NaN]] %[[#NaN]] %[[#NaN]]

; CHECK-LABEL: Begin function minimum_bf16
; CHECK: %[[#A:]] = OpFunctionParameter %[[#BF16]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#BF16]]
; CHECK: %[[#M:]] = OpExtInst %[[#BF16]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; CHECK: %[[#ASign:]] = OpExtInst %[[#BF16]] %[[#Ext]] copysign %[[#One]] %[[#A]]
; CHECK: %[[#ANeg:]] = OpFOrdLessThan %[[#]] %[[#ASign]] %[[#]]
; CHECK: %[[#Sign:]] = OpSelect %[[#BF16]] %[[#ANeg]] %[[#A]] %[[#B]]
; CHECK: %[[#Signed:]] = OpExtInst %[[#BF16]] %[[#Ext]] copysign %[[#M]] %[[#Sign]]
; CHECK: %[[#Uno:]] = OpUnordered %[[#]] %[[#A]] %[[#B]]
; CHECK: %[[#Res:]] = OpSelect %[[#BF16]] %[[#Uno]] %[[#NaN]] %[[#Signed]]
; CHECK: OpReturnValue %[[#Res]]
define bfloat @minimum_bf16(bfloat %a, bfloat %b) {
  %r = call bfloat @llvm.minimum.bf16(bfloat %a, bfloat %b)
  ret bfloat %r
}

; CHECK-LABEL: Begin function maximum_v4bf16
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4BF16]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#V4BF16]]
; CHECK: %[[#M:]] = OpExtInst %[[#V4BF16]] %[[#Ext]] fmax %[[#A]] %[[#B]]
; CHECK: %[[#ASign:]] = OpExtInst %[[#V4BF16]] %[[#Ext]] copysign %[[#V4One]] %[[#A]]
; CHECK: %[[#ANeg:]] = OpFOrdLessThan %[[#]] %[[#ASign]] %[[#]]
; CHECK: %[[#Sign:]] = OpSelect %[[#V4BF16]] %[[#ANeg]] %[[#B]] %[[#A]]
; CHECK: %[[#Signed:]] = OpExtInst %[[#V4BF16]] %[[#Ext]] copysign %[[#M]] %[[#Sign]]
; CHECK: %[[#Uno:]] = OpUnordered %[[#]] %[[#A]] %[[#B]]
; CHECK: %[[#Res:]] = OpSelect %[[#V4BF16]] %[[#Uno]] %[[#V4NaN]] %[[#Signed]]
; CHECK: OpReturnValue %[[#Res]]
define <4 x bfloat> @maximum_v4bf16(<4 x bfloat> %a, <4 x bfloat> %b) {
  %r = call <4 x bfloat> @llvm.maximum.v4bf16(<4 x bfloat> %a, <4 x bfloat> %b)
  ret <4 x bfloat> %r
}

; Reductions on vectors whose length is not a power of two are expanded into a
; chain of llvm.minimum.
; CHECK-LABEL: Begin function reduce_fminimum_v3bf16
; CHECK: %[[#V:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#E0:]] = OpCompositeExtract %[[#BF16]] %[[#V]] 0
; CHECK: %[[#E1:]] = OpCompositeExtract %[[#BF16]] %[[#V]] 1
; CHECK: %[[#E2:]] = OpCompositeExtract %[[#BF16]] %[[#V]] 2
; CHECK: %[[#M1:]] = OpExtInst %[[#BF16]] %[[#Ext]] fmin %[[#E0]] %[[#E1]]
; CHECK: OpExtInst %[[#BF16]] %[[#Ext]] copysign %[[#One]] %[[#E0]]
; CHECK: %[[#R1:]] = OpSelect %[[#BF16]] %[[#]] %[[#NaN]] %[[#]]
; CHECK: %[[#M2:]] = OpExtInst %[[#BF16]] %[[#Ext]] fmin %[[#R1]] %[[#E2]]
; CHECK: OpExtInst %[[#BF16]] %[[#Ext]] copysign %[[#One]] %[[#R1]]
; CHECK: %[[#R2:]] = OpSelect %[[#BF16]] %[[#]] %[[#NaN]] %[[#]]
; CHECK: OpReturnValue %[[#R2]]
define bfloat @reduce_fminimum_v3bf16(<3 x bfloat> %v) {
  %r = call bfloat @llvm.vector.reduce.fminimum.v3bf16(<3 x bfloat> %v)
  ret bfloat %r
}
