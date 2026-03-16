; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%struct.simple = type { i8 }
@g_simple = private unnamed_addr addrspace(2) constant %struct.simple poison, align 1

%struct.multi = type { i32, float, i8 }
@g_multi = private addrspace(2) constant %struct.multi poison, align 4

@g_arr = private addrspace(2) constant [3 x i32] poison, align 4

%struct.inner = type { i32 }
%struct.outer = type { %struct.inner, float }
@g_nested = private addrspace(2) constant %struct.outer poison, align 4

define spir_kernel void @k() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0

; CHECK-DAG: %[[#INNER:]] = OpTypeStruct %[[#I32]]
; CHECK-DAG: %[[#OUTER:]] = OpTypeStruct %[[#INNER]] %[[#F32]]
; CHECK-DAG: %[[#MULTI:]] = OpTypeStruct %[[#I32]] %[[#F32]] %[[#I8]]
; CHECK-DAG: %[[#SIMPLE:]] = OpTypeStruct %[[#I8]]
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]

; CHECK-DAG: %[[#OUTER_PTR:]] = OpTypePointer UniformConstant %[[#OUTER]]
; CHECK-DAG: %[[#ARR_PTR:]] = OpTypePointer UniformConstant %[[#ARR]]
; CHECK-DAG: %[[#MULTI_PTR:]] = OpTypePointer UniformConstant %[[#MULTI]]
; CHECK-DAG: %[[#SIMPLE_PTR:]] = OpTypePointer UniformConstant %[[#SIMPLE]]

; CHECK: OpConstantComposite %[[#SIMPLE]]
; CHECK: OpVariable %[[#SIMPLE_PTR]] UniformConstant
; CHECK: OpConstantComposite %[[#MULTI]]
; CHECK: OpVariable %[[#MULTI_PTR]] UniformConstant
; CHECK: OpConstantComposite %[[#ARR]]
; CHECK: OpVariable %[[#ARR_PTR]] UniformConstant
; CHECK: OpConstantComposite %[[#OUTER]]
; CHECK: OpVariable %[[#OUTER_PTR]] UniformConstant

!0 = !{}
