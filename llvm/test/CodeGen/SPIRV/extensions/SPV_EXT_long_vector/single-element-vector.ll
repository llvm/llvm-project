; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; spirv-val has a bug around validating OpTypeVectorIdEXTs with Pointer type elements, as it seems to only allow scalar numerical types for the latter.
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FLOAT64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#INT8]]
; CHECK-DAG: %[[#VEC4:]] = OpTypeVector %[[#FLOAT64]] 4
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#INT32]] 1
; CHECK-DAG: %[[#VEC1:]] = OpTypeVectorIdEXT %[[#INT16]] %[[#ONE]]
; CHECK-DAG: %[[#PVEC1:]] = OpTypeVectorIdEXT %[[#PTR]] %[[#ONE]]
; CHECK-DAG: %[[#PVEC1P:]] = OpTypePointer CrossWorkgroup %[[#PVEC1]]
; CHECK-DAG: %[[#FVEC1:]] = OpTypeVectorIdEXT %[[#FLOAT64]] %[[#ONE]]
; CHECK-DAG: %[[#FNTY:]] = OpTypeFunction %[[#VEC4]] %[[#VEC1]]
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#VEC1]]
; CHECK-DAG: %[[#NULLPTR:]] = OpConstantNull %[[#PVEC1]]
; CHECK-DAG: %[[#ISPLAT42:]] = OpConstantComposite %[[#VEC1]]
; CHECK-DAG: %[[#FSPLAT42:]] = OpConstantComposite %[[#FVEC1]]

; CHECK: OpFunctionCall %[[#VEC4]] %[[#]] %[[#ZERO]]
define spir_func <4 x double> @caller() {
entry:
  %C = call <4 x double> @callee(<1 x i16> zeroinitializer)
  ret <4 x double> %C
}
declare <4 x double> @callee(<1 x i16>)

; CHECK: %[[#V:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: %[[#EXTRACT_RES:]] = OpCompositeExtract %[[#INT16]] %[[#V]] 0
; CHECK: OpReturnValue %[[#EXTRACT_RES]]
define spir_func i16 @test_extractelement(<1 x i16> %v) {
entry:
  %e = extractelement <1 x i16> %v, i32 0
  ret i16 %e
}

; CHECK: %[[#VAL:]] = OpFunctionParameter %[[#INT16]]
; CHECK: %[[#INSERT_VAL:]] = OpCompositeInsert %[[#VEC1]] %[[#VAL]] %[[#]] 0
; CHECK: OpReturnValue %[[#INSERT_VAL]]
define spir_func <1 x i16> @test_insertelement(i16 %val) {
entry:
  %v = insertelement <1 x i16> poison, i16 %val, i32 0
  ret <1 x i16> %v
}

; CHECK: %[[#SHUF_PARAM:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: OpReturnValue %[[#SHUF_PARAM]]
define spir_func <1 x i16> @test_shufflevector(<1 x i16> %v) {
entry:
  %s = shufflevector <1 x i16> %v, <1 x i16> poison, <1 x i32> zeroinitializer
  ret <1 x i16> %s
}

; CHECK: %[[#LHS_PARAM:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: %[[#RHS_PARAM:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: %[[#RET:]] = OpIAdd %[[#VEC1]] %[[#LHS_PARAM]] %[[#RHS_PARAM]]
; CHECK: OpReturnValue %[[#RET]]
define spir_func <1 x i16> @test_arithm(<1 x i16> %v1, <1 x i16> %v2) {
entry:
  %s = add <1 x i16> %v1, %v2
  ret <1 x i16> %s
}

; CHECK: OpReturnValue %[[#ISPLAT42]]
define <1 x i16> @explicit_int_splat() {
  ret <1 x i16> splat (i16 42)
}

; CHECK: OpReturnValue %[[#FSPLAT42]]
define <1 x double> @explicit_float_splat() {
  ret <1 x double> splat (double 42.0)
}

; CHECK: OpReturnValue %[[#NULLPTR]]
define <1 x ptr addrspace(1)> @explicit_fixed_ptr_null_splat() {
  ret <1 x ptr addrspace(1)> splat (ptr addrspace(1) null)
}

; CHECK: %[[#OUT:]] = OpFunctionParameter %[[#PVEC1P]]
; CHECK: OpStore %[[#OUT]] %[[#NULLPTR]]
define void @store_fixed_ptr_null_splat(ptr addrspace(1) %out) {
  store <1 x ptr addrspace(1)> splat (ptr addrspace(1) null), ptr addrspace(1) %out
  ret void
}
