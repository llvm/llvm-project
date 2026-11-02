; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,OPENCL,PHYSICAL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,OPENCL,PHYSICAL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,OPENCL,PHYSICAL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED,OPENCL --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED,SHADER --implicit-check-not=VariableLengthArrayINTEL --implicit-check-not=SPV_INTEL_variable_length_array
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -filetype=obj -o - | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -filetype=obj -o - | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_untyped_pointers %s -filetype=obj -o - | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -filetype=obj -o - | spirv-val %}

; Function-storage allocas with constant element counts use array variables,
; even when the VLA extension is enabled.

; CHECK-DAG: OpName %[[#SCALARS:]] "scalars"
; CHECK-DAG: OpName %[[#STRUCTS:]] "structs"
; OPENCL-DAG: OpDecorate %[[#SCALARS]] Alignment 16
; OPENCL-DAG: OpDecorate %[[#STRUCTS]] Alignment 8
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#INT]] 1{{$}}
; SHADER-DAG: %[[#ZERO:]] = OpConstant %[[#INT]] 0{{$}}
; CHECK-DAG: %[[#THREE:]] = OpConstant %[[#INT]] 3{{$}}
; CHECK-DAG: %[[#FOUR:]] = OpConstant %[[#INT]] 4{{$}}
; CHECK-DAG: %[[#INT_ARRAY:]] = OpTypeArray %[[#INT]] %[[#FOUR]]
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#INT]] %[[#INT]]
; CHECK-DAG: %[[#STRUCT_ARRAY:]] = OpTypeArray %[[#STRUCT]] %[[#THREE]]
; TYPED-DAG: %[[#INT_PTR:]] = OpTypePointer Function %[[#INT]]
; PHYSICAL-DAG: %[[#STRUCT_PTR:]] = OpTypePointer Function %[[#STRUCT]]
; TYPED-DAG: %[[#INT_ARRAY_PTR:]] = OpTypePointer Function %[[#INT_ARRAY]]
; TYPED-DAG: %[[#STRUCT_ARRAY_PTR:]] = OpTypePointer Function %[[#STRUCT_ARRAY]]
; UNTYPED-DAG: %[[#PTR:]] = OpTypeUntypedPointerKHR Function

; CHECK-LABEL: Begin function scalar_count
; CHECK: %[[#INDEX:]] = OpFunctionParameter %[[#INT]]
; TYPED: %[[#SCALARS]] = OpVariable %[[#INT_ARRAY_PTR]] Function
; UNTYPED: %[[#SCALARS]] = OpUntypedVariableKHR %[[#PTR]] Function %[[#INT_ARRAY]]
; PHYSICAL: %[[#BASE:]] = OpBitcast %[[#INT_PTR]] %[[#SCALARS]]
; SHADER: %[[#BASE:]] = OpInBoundsAccessChain %[[#INT_PTR]] %[[#SCALARS]] %[[#ZERO]]
; TYPED: OpStore %[[#BASE]]
; UNTYPED: OpStore %[[#SCALARS]]
; PHYSICAL: %[[#ELEMENT:]] = OpInBoundsPtrAccessChain %[[#INT_PTR]] %[[#BASE]] %[[#INDEX]]
; SHADER: %[[#ELEMENT:]] = OpInBoundsAccessChain %[[#INT_PTR]] %[[#SCALARS]] %[[#INDEX]]
; UNTYPED: %[[#ELEMENT:]] = OpUntypedInBoundsPtrAccessChainKHR %[[#PTR]] %[[#INT]] %[[#SCALARS]] %[[#INDEX]]
; CHECK: OpStore %[[#ELEMENT]]
; PHYSICAL: OpLoad %[[#INT]] %[[#BASE]]
; UNTYPED: OpLoad %[[#INT]] %[[#SCALARS]]
; SHADER: %[[#RELOAD:]] = OpAccessChain %[[#INT_PTR]] %[[#SCALARS]] %[[#ZERO]]
; SHADER: OpLoad %[[#INT]] %[[#RELOAD]]
define spir_func i32 @scalar_count(i32 %index) {
entry:
  %scalars = alloca i32, i32 4, align 16
  store volatile i32 7, ptr %scalars, align 4
  %element = getelementptr inbounds i32, ptr %scalars, i32 %index
  store volatile i32 9, ptr %element, align 4
  %value = load volatile i32, ptr %scalars, align 4
  ret i32 %value
}

; CHECK-LABEL: Begin function struct_count
; CHECK: %[[#INDEX:]] = OpFunctionParameter %[[#INT]]
; TYPED: %[[#STRUCTS]] = OpVariable %[[#STRUCT_ARRAY_PTR]] Function
; UNTYPED: %[[#STRUCTS]] = OpUntypedVariableKHR %[[#PTR]] Function %[[#STRUCT_ARRAY]]
; PHYSICAL: %[[#BASE:]] = OpBitcast %[[#STRUCT_PTR]] %[[#STRUCTS]]
; PHYSICAL: %[[#FIELD:]] = OpInBoundsPtrAccessChain %[[#INT_PTR]] %[[#BASE]] %[[#INDEX]] %[[#ONE]]
; SHADER: %[[#FIELD:]] = OpInBoundsAccessChain %[[#INT_PTR]] %[[#STRUCTS]] %[[#INDEX]] %[[#ONE]]
; UNTYPED: %[[#FIELD:]] = OpUntypedInBoundsPtrAccessChainKHR %[[#PTR]] %[[#STRUCT]] %[[#STRUCTS]] %[[#INDEX]] %[[#ONE]]
; CHECK: OpStore %[[#FIELD]]
; CHECK: OpLoad %[[#INT]] %[[#FIELD]]
define spir_func i32 @struct_count(i32 %index) {
entry:
  %structs = alloca { i32, i32 }, i64 3, align 8
  %field = getelementptr inbounds { i32, i32 }, ptr %structs, i32 %index, i32 1
  store volatile i32 11, ptr %field, align 4
  %value = load volatile i32, ptr %field, align 4
  ret i32 %value
}
