; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --check-prefix=CHECK-HLSL
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-OCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define i32 @getNullIntConstant() {
  ret i32 0
}

define float @getNullFloatConstant() {
  ret float +0.0
}

define <4 x i32> @getNullIntVectorConstant() {
  ret <4 x i32> splat (i32 0)
}

define <4 x float> @getNullFloatVectorConstant() {
  ret <4 x float> splat (float +0.0)
}

define [2 x <4 x i32>] @getNullIntArrayOfVectorConstant() {
  ret [2 x <4 x i32>] [ <4 x i32> splat (i32 0), <4 x i32> splat (i32 0) ]
}

define [2 x <4 x float>] @getNullFloatArrayOfVectorConstant() {
  ret [2 x <4 x float>] [ <4 x float> splat (float +0.0), <4 x float> splat (float +0.0) ]
}

%null.struct = type { i32, float, <4 x i32>, <4 x float> }
define %null.struct @getNullStructConstant() {
  ret %null.struct { i32 0, float +0.0, <4 x i32> splat (i32 0), <4 x float> splat (float +0.0) }
}

; CHECK-HLSL-DAG: [[I32:%.+]] = OpTypeInt 32 0
; CHECK-HLSL-DAG: [[F32:%.+]] = OpTypeFloat 32
; CHECK-HLSL-DAG: [[INT_VECTOR:%.+]] = OpTypeVector [[I32]]
; CHECK-HLSL-DAG: [[FLOAT_VECTOR:%.+]] = OpTypeVector [[F32]]
; CHECK-HLSL-DAG: [[INT_VECTOR_ARRAY:%.+]] = OpTypeArray [[INT_VECTOR]]
; CHECK-HLSL-DAG: [[FLOAT_VECTOR_ARRAY:%.+]] = OpTypeArray [[FLOAT_VECTOR]]
; CHECK-HLSL-DAG: [[NULL_STRUCT:%.+]] = OpTypeStruct [[I32]] [[F32]] [[INT_VECTOR]] [[FLOAT_VECTOR]]
; CHECK-HLSL-DAG: [[NULL_F32:%.+]] = OpConstant [[F32]] 0
; CHECK-HLSL-DAG: [[NULL_INT_VECTOR:%.+]] = OpConstantNull [[INT_VECTOR]]
; CHECK-HLSL-DAG: [[NULL_FLOAT_VECTOR:%.+]] = OpConstantComposite [[FLOAT_VECTOR]] [[NULL_F32]] [[NULL_F32]] [[NULL_F32]] [[NULL_F32]]
; CHECK-HLSL-DAG: [[NULL_INT_VECTOR_ARRAY:%.+]] = OpConstantNull [[INT_VECTOR_ARRAY]]
; CHECK-HLSL-DAG: [[NULL_FLOAT_VECTOR_ARRAY:%.+]] = OpConstantNull [[FLOAT_VECTOR_ARRAY]]
; CHECK-HLSL-DAG: [[NULL_STRUCT_CONST:%.+]] = OpConstantNull [[NULL_STRUCT]]

; CHECK-OCL-DAG: [[I32:%.+]] = OpTypeInt 32
; CHECK-OCL-DAG: [[F32:%.+]] = OpTypeFloat 32
; CHECK-OCL-DAG: [[INT_VECTOR:%.+]] = OpTypeVector [[I32]]
; CHECK-OCL-DAG: [[FLOAT_VECTOR:%.+]] = OpTypeVector [[F32]]
; CHECK-OCL-DAG: [[INT_VECTOR_ARRAY:%.+]] = OpTypeArray [[INT_VECTOR]]
; CHECK-OCL-DAG: [[FLOAT_VECTOR_ARRAY:%.+]] = OpTypeArray [[FLOAT_VECTOR]]
; CHECK-OCL-DAG: [[NULL_STRUCT:%.+]] = OpTypeStruct [[I32]] [[F32]] [[INT_VECTOR]] [[FLOAT_VECTOR]]
; CHECK-OCL-DAG: [[NULL_I32:%.+]] = OpConstantNull [[I32]]
; CHECK-OCL-DAG: [[NULL_F32:%.+]] = OpConstantNull [[F32]]
; CHECK-OCL-DAG: [[NULL_INT_VECTOR:%.+]] = OpConstantNull [[INT_VECTOR]]
; CHECK-OCL-DAG: [[NULL_FLOAT_VECTOR:%.+]] = OpConstantComposite [[FLOAT_VECTOR]] [[NULL_F32]] [[NULL_F32]] [[NULL_F32]] [[NULL_F32]]
; CHECK-OCL-DAG: [[NULL_INT_VECTOR_ARRAY:%.+]] = OpConstantNull [[INT_VECTOR_ARRAY]]
; CHECK-OCL-DAG: [[NULL_FLOAT_VECTOR_ARRAY:%.+]] = OpConstantNull [[FLOAT_VECTOR_ARRAY]]
; CHECK-OCL-DAG: [[NULL_STRUCT_CONST:%.+]] = OpConstantNull [[NULL_STRUCT]]
