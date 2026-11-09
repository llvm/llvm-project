; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefix=CHECK-PHYSICAL
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefix=CHECK-LOGICAL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; A GEP whose only index is zero stays an access chain. The zero is the Element
; operand in physical SPIR-V, and in logical SPIR-V there is no Element operand,
; so the instruction is emitted with no Indexes at all.

%struct.S = type { i32, float }

; CHECK-PHYSICAL-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-PHYSICAL-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-PHYSICAL-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-PHYSICAL-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-PHYSICAL-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#F32]]
; CHECK-PHYSICAL-DAG: %[[#NULL64:]] = OpConstantNull %[[#I64]]
; CHECK-PHYSICAL-DAG: %[[#CONST1_32:]] = OpConstant %[[#I32]] 1

; CHECK-LOGICAL-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-LOGICAL-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-LOGICAL-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-LOGICAL-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#F32]]
; CHECK-LOGICAL-DAG: %[[#CONST1_32:]] = OpConstant %[[#I32]] 1

; CHECK-PHYSICAL: OpFunction
; CHECK-PHYSICAL: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK-PHYSICAL: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#P]] %[[#NULL64]]

; CHECK-LOGICAL: OpFunction
; CHECK-LOGICAL: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK-LOGICAL: %[[#]] = OpUntypedAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#P]]{{$}}
define spir_kernel void @gep_zero_only(ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %q = getelementptr i32, ptr addrspace(1) %p, i64 0
  %v = load i32, ptr addrspace(1) %q, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-PHYSICAL: OpFunction
; CHECK-PHYSICAL: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK-PHYSICAL: %[[#]] = OpUntypedInBoundsPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#P]] %[[#NULL64]]

; CHECK-LOGICAL: OpFunction
; CHECK-LOGICAL: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK-LOGICAL: %[[#]] = OpUntypedInBoundsAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#P]]{{$}}
define spir_kernel void @gep_zero_only_inbounds(ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 0
  %v = load i32, ptr addrspace(1) %q, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

; The logical form drops the leading zero, so only the field index survives.
; CHECK-PHYSICAL: OpFunction
; CHECK-PHYSICAL: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK-PHYSICAL: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#STRUCT]] %[[#P]] %[[#NULL64]] %[[#CONST1_32]]

; CHECK-LOGICAL: OpFunction
; CHECK-LOGICAL: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK-LOGICAL: %[[#]] = OpUntypedAccessChainKHR %[[#CROSS_PTR]] %[[#STRUCT]] %[[#P]] %[[#CONST1_32]]
define spir_kernel void @gep_zero_then_field(ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %q = getelementptr %struct.S, ptr addrspace(1) %p, i64 0, i32 1
  %v = load float, ptr addrspace(1) %q, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}
