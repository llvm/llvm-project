; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; We can't add the validator until we support VUID-StandaloneSpirv-Location-04916
; TODO %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpName %[[#INT_VAR:]] "int_var"
; CHECK-DAG: OpName %[[#DOUBLE_VAR:]] "double_var"
; CHECK-DAG: OpName %[[#VEC_INT_VAR:]] "vec_int_var"
; CHECK-DAG: OpName %[[#ARR_DOUBLE_VAR:]] "arr_double_var"
; CHECK-DAG: OpName %[[#ARR_VEC_INT_VAR:]] "arr_vec_int_var"
; CHECK-DAG: OpName %[[#FLOAT_VAR:]] "float_var"
; CHECK-DAG: OpName %[[#PRIVATE_INT_VAR:]] "private_int_var"

; -----------------------------------------------------------------------------
; 1. Verify int and doubles scalar and element types receive the Flat decoration.
; -----------------------------------------------------------------------------
; CHECK-DAG: OpDecorate %[[#INT_VAR]] Flat
; CHECK-DAG: OpDecorate %[[#DOUBLE_VAR]] Flat
; CHECK-DAG: OpDecorate %[[#VEC_INT_VAR]] Flat
; CHECK-DAG: OpDecorate %[[#ARR_DOUBLE_VAR]] Flat
; CHECK-DAG: OpDecorate %[[#ARR_VEC_INT_VAR]] Flat

; -----------------------------------------------------------------------------
; 2. Verify that the negative test cases DO NOT receive the Flat decoration.
; -----------------------------------------------------------------------------
; CHECK-NOT: OpDecorate %[[#FLOAT_VAR]] Flat
; CHECK-NOT: OpDecorate %[[#PRIVATE_INT_VAR]] Flat

; scalar flat cases
@int_var = internal addrspace(7) global i32 poison
@double_var = internal addrspace(7) global double poison

; vector flat case
@vec_int_var = internal addrspace(7) global <4 x i32> poison

; array flat case
@arr_double_var = internal addrspace(7) global [4 x double] poison

; array of vector flat case
@arr_vec_int_var = internal addrspace(7) global [2 x <4 x i32>] poison

; float case so no Flat decoration
@float_var = internal addrspace(7) global float poison

; wrong addresspace so no Flat decoration
@private_int_var = internal addrspace(0) global i32 poison

define void @main() #1 {
entry:
  %0 = load volatile i32, ptr addrspace(7) @int_var
  %1 = load volatile double, ptr addrspace(7) @double_var
  %2 = load volatile <4 x i32>, ptr addrspace(7) @vec_int_var
  %3 = load volatile [4 x double], ptr addrspace(7) @arr_double_var
  %4 = load volatile [2 x <4 x i32>], ptr addrspace(7) @arr_vec_int_var
  
  %5 = load volatile float, ptr addrspace(7) @float_var
  %6 = load volatile i32, ptr addrspace(0) @private_int_var
  ret void
}

attributes #1 = { "hlsl.shader"="pixel" }

