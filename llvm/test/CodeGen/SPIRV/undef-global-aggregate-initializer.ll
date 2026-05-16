; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%struct.simple = type { i8 }
@g_simple = private unnamed_addr addrspace(2) constant %struct.simple poison, align 1

%struct.multi = type { i32, float, i8 }
@g_multi = private addrspace(2) constant %struct.multi poison, align 4

@g_arr = private addrspace(2) constant [3 x i32] poison, align 4

%struct.inner = type { i32 }
%struct.outer = type { %struct.inner, float }
@g_nested = private addrspace(2) constant %struct.outer poison, align 4

%struct.mixed = type { i32, float }
@g_mixed = private addrspace(2) constant %struct.mixed { i32 poison, float 1.0 }, align 4

%struct.with_arr = type { [2 x i32], float }
@g_struct_with_arr = private addrspace(2) constant %struct.with_arr poison, align 4

@g_arr_of_struct = private addrspace(2) constant [2 x %struct.with_arr] poison, align 4

define spir_func void @foo() {
entry:
  ret void
}

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I32_ARR2:]] = OpTypeArray %[[#I32]] %[[#]]
; CHECK-DAG: %[[#MULTI:]] = OpTypeStruct %[[#I32]] %[[#F32]] %[[#I8]]
; CHECK-DAG: %[[#SIMPLE:]] = OpTypeStruct %[[#I8]]{{$}}
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]
; CHECK-DAG: %[[#INNER:]] = OpTypeStruct %[[#I32]]{{$}}
; CHECK-DAG: %[[#WITH_ARR:]] = OpTypeStruct %[[#I32_ARR2]] %[[#F32]]

; CHECK-DAG: %[[#OUTER:]] = OpTypeStruct %[[#INNER]] %[[#F32]]
; CHECK-DAG: %[[#MIXED:]] = OpTypeStruct %[[#I32]] %[[#F32]]{{$}}
; CHECK-DAG: %[[#ARR_OF_STRUCT:]] = OpTypeArray %[[#WITH_ARR]] %[[#]]

; CHECK-DAG: %[[#ARR_OF_STRUCT_PTR:]] = OpTypePointer UniformConstant %[[#ARR_OF_STRUCT]]
; CHECK-DAG: %[[#WITH_ARR_PTR:]] = OpTypePointer UniformConstant %[[#WITH_ARR]]
; CHECK-DAG: %[[#MIXED_PTR:]] = OpTypePointer UniformConstant %[[#MIXED]]
; CHECK-DAG: %[[#OUTER_PTR:]] = OpTypePointer UniformConstant %[[#OUTER]]
; CHECK-DAG: %[[#ARR_PTR:]] = OpTypePointer UniformConstant %[[#ARR]]
; CHECK-DAG: %[[#MULTI_PTR:]] = OpTypePointer UniformConstant %[[#MULTI]]
; CHECK-DAG: %[[#SIMPLE_PTR:]] = OpTypePointer UniformConstant %[[#SIMPLE]]

; CHECK-DAG: %[[#CONST_F32:]] = OpConstant %[[#F32]] 1
; CHECK-DAG: %[[#UNDEF_I8:]] = OpUndef %[[#I8]]
; CHECK-DAG: %[[#UNDEF_I32:]] = OpUndef %[[#I32]]
; CHECK-DAG: %[[#UNDEF_F32:]] = OpUndef %[[#F32]]
; CHECK-DAG: %[[#UNDEF_INNER:]] = OpUndef %[[#INNER]]
; CHECK-DAG: %[[#UNDEF_I32_ARR2:]] = OpUndef %[[#I32_ARR2]]
; CHECK-DAG: %[[#UNDEF_WITH_ARR:]] = OpUndef %[[#WITH_ARR]]

; CHECK-DAG: OpConstantComposite %[[#SIMPLE]] %[[#UNDEF_I8]]
; CHECK-DAG: OpVariable %[[#SIMPLE_PTR]] UniformConstant
; CHECK-DAG: OpConstantComposite %[[#MULTI]] %[[#UNDEF_I32]] %[[#UNDEF_F32]] %[[#UNDEF_I8]]
; CHECK-DAG: OpVariable %[[#MULTI_PTR]] UniformConstant
; CHECK-DAG: OpConstantComposite %[[#ARR]] %[[#UNDEF_I32]] %[[#UNDEF_I32]] %[[#UNDEF_I32]]
; CHECK-DAG: OpVariable %[[#ARR_PTR]] UniformConstant
; CHECK-DAG: OpConstantComposite %[[#OUTER]] %[[#UNDEF_INNER]] %[[#UNDEF_F32]]
; CHECK-DAG: OpVariable %[[#OUTER_PTR]] UniformConstant
; CHECK-DAG: OpConstantComposite %[[#MIXED]] %[[#UNDEF_I32]] %[[#CONST_F32]]
; CHECK-DAG: OpVariable %[[#MIXED_PTR]] UniformConstant
; CHECK-DAG: OpConstantComposite %[[#WITH_ARR]] %[[#UNDEF_I32_ARR2]] %[[#UNDEF_F32]]
; CHECK-DAG: OpVariable %[[#WITH_ARR_PTR]] UniformConstant
; CHECK-DAG: OpConstantComposite %[[#ARR_OF_STRUCT]] %[[#UNDEF_WITH_ARR]] %[[#UNDEF_WITH_ARR]]
; CHECK-DAG: OpVariable %[[#ARR_OF_STRUCT_PTR]] UniformConstant
