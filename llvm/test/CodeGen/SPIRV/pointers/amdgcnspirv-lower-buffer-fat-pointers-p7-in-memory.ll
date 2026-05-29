; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

target triple = "spirv64-amd-amdhsa"

; CHECK: OpName %[[#SCALAR_COPY:]] "scalar_copy"
; CHECK: OpName %[[#VECTOR_COPY:]] "vector_copy"
; CHECK: OpName %[[#ALLOCA:]] "alloca"
; CHECK: OpName %[[#COMPLEX_COPY:]] "complex_copy"
; CHECK: %[[#INT8_TY:]] = OpTypeInt 8 0
; CHECK: %[[#I8PTR_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#INT8_TY]]
; CHECK: %[[#I8PTR_ADDRSPACE_7PTR_GENERIC:]] = OpTypePointer Generic %[[#I8PTR_ADDRSPACE_7]]
; CHECK: %[[#VEC4_I8PTR_ADDRSPACE_7:]] = OpTypeVector %[[#I8PTR_ADDRSPACE_7]] 4
; CHECK: %[[#VEC4_I8PTR_ADDRSPACE_7PTR_GENERIC:]] = OpTypePointer Generic %[[#VEC4_I8PTR_ADDRSPACE_7]]
; CHECK: %[[#I8PTR_ADDRSPACE_7PTR_PRIVATE:]] = OpTypePointer Function %[[#I8PTR_ADDRSPACE_7]]

; CHECK: %[[#SCALAR_COPY]] = OpFunction
; CHECK: %[[#SCALAR_COPY_A:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_7PTR_GENERIC]]
; CHECK: %[[#SCALAR_COPY_B:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_7PTR_GENERIC]]
; CHECK: %[[#SCALAR_COPY_X:]] = OpLoad %[[#I8PTR_ADDRSPACE_7]] %[[#SCALAR_COPY_A]]
; CHECK: %[[#SCALAR_COPY_B1:]] = OpPtrAccessChain %[[#I8PTR_ADDRSPACE_7PTR_GENERIC]] %[[#SCALAR_COPY_B]]
; CHECK: OpStore %[[#SCALAR_COPY_B1]] %[[#SCALAR_COPY_X]]
define void @scalar_copy(ptr addrspace(4) %a, ptr addrspace(4) %b) {
  %x = load ptr addrspace(7), ptr addrspace(4) %a
  %b1 = getelementptr ptr addrspace(7), ptr addrspace(4) %b, i64 1
  store ptr addrspace(7) %x, ptr addrspace(4) %b1
  ret void
}

; CHECK: %[[#VECTOR_COPY:]] = OpFunction
; CHECK: %[[#VECTOR_COPY_A:]] = OpFunctionParameter %[[#VEC4_I8PTR_ADDRSPACE_7PTR_GENERIC]]
; CHECK: %[[#VECTOR_COPY_B:]] = OpFunctionParameter %[[#VEC4_I8PTR_ADDRSPACE_7PTR_GENERIC]]
; CHECK: %[[#VECTOR_COPY_X:]] = OpLoad %[[#VEC4_I8PTR_ADDRSPACE_7]] %[[#VECTOR_COPY_A]]
; CHECK: %[[#VECTOR_COPY_B1:]] = OpPtrAccessChain %[[#VEC4_I8PTR_ADDRSPACE_7PTR_GENERIC]] %[[#VECTOR_COPY_B]]
; CHECK: %[[#VECTOR_COPY_B1_BC:]] = OpBitcast %[[#VEC4PTR_VEC4_I8PTR_ADDRSPACE_7:]] %[[#VECTOR_COPY_B1]]
; CHECK: %[[#VECTOR_COPY_B1_BC_BC:]] = OpBitcast %[[#VEC4_I8PTR_ADDRSPACE_7PTR_GENERIC]] %[[#VECTOR_COPY_B1_BC]]
; CHECK: OpStore %[[#VECTOR_COPY_B1_BC_BC]] %[[#VECTOR_COPY_X]]
define void @vector_copy(ptr addrspace(4) %a, ptr addrspace(4) %b) {
  %x = load <4 x ptr addrspace(7)>, ptr addrspace(4) %a
  %b1 = getelementptr <4 x ptr addrspace(7)>, ptr addrspace(4) %b, i64 2
  store <4 x ptr addrspace(7)> %x, ptr addrspace(4) %b1
  ret void
}

; CHECK: %[[#ALLOCA]] = OpFunction
; CHECK: %[[#ALLOCA_A:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_7PTR_GENERIC]]
; CHECK: %[[#ALLOCA_B:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_7PTR_GENERIC]]
; CHECK: %[[#ALLOCA_ARR:]] = OpVariable %[[#ARR5_I8PTR_ADDRSPACE_7:]] Function
; CHECK: %[[#ALLOCA_X:]] = OpLoad %[[#I8PTR_ADDRSPACE_7]] %[[#ALLOCA_A]]
; CHECK: %[[#ALLOCA_ARR_BC:]] = OpBitcast %[[#I8PTR_ADDRSPACE_7PTR_PRIVATE]] %[[#ALLOCA_ARR]]
; CHECK: %[[#ALLOCA_L:]] = OpPtrAccessChain %[[#I8PTR_ADDRSPACE_7PTR_PRIVATE]] %[[#ALLOCA_ARR_BC]] %[[#]]
; CHECK: OpStore %[[#ALLOCA_L]] %[[#ALLOCA_X]]
; CHECK: %[[#ALLOCA_Y:]] = OpLoad %[[#I8PTR_ADDRSPACE_7]] %[[#ALLOCA_L]]
; CHECK: OpStore %[[#ALLOCA_B]] %[[#ALLOCA_Y]]
define void @alloca(ptr addrspace(4) %a, ptr addrspace(4) %b) {
  %alloca = alloca [5 x ptr addrspace(7)]
  %x = load ptr addrspace(7), ptr addrspace(4) %a
  %l = getelementptr ptr addrspace(7), ptr %alloca, i32 1
  store ptr addrspace(7) %x, ptr %l
  %y = load ptr addrspace(7), ptr %l
  store ptr addrspace(7) %y, ptr addrspace(4) %b
  ret void
}

; CHECK: %[[#COMPLEX_COPY]] = OpFunction
; CHECK: %[[#COMPLEX_COPY_A:]] = OpFunctionParameter %[[#STRUCT1_PTR_TY:]]
; CHECK: %[[#COMPLEX_COPY_B:]] = OpFunctionParameter %[[#WRAPPED_STRUCT1_PTR_TY:]]
; CHECK: %[[#COMPLEX_COPY_X:]] = OpLoad %[[#STRUCT1_TY:]] %[[#COMPLEX_COPY_A]]
; CHECK: %[[#COMPLEX_COPY_B_BC:]] = OpBitcast %[[#STRUCT1_PTR_TY]] %[[#COMPLEX_COPY_B]]
; CHECK: OpStore %[[#COMPLEX_COPY_B_BC]] %[[#COMPLEX_COPY_X]]
define void @complex_copy(ptr addrspace(4) %a, ptr addrspace(4) %b) {
  %x = load {[2 x ptr addrspace(7)], i32, ptr addrspace(7)}, ptr addrspace(4) %a
  store {[2 x ptr addrspace(7)], i32, ptr addrspace(7)} %x, ptr addrspace(4) %b
  ret void
}
