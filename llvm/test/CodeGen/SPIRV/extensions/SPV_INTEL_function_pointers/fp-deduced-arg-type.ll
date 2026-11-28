; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; The type of a function pointer must point to the type of the OpFunction, which
; uses the deduced pointee types of pointer arguments, both for the function
; that is being lowered and for a function that is lowered later.

; CHECK-DAG: %[[#VTable:]] = OpTypeStruct %[[#]]
; CHECK-DAG: %[[#VTablePtr:]] = OpTypePointer CrossWorkgroup %[[#VTable]]
; CHECK-DAG: %[[#ThisPtr:]] = OpTypePointer Generic %[[#VTablePtr]]
; CHECK-DAG: %[[#FunTy:]] = OpTypeFunction %[[#]] %[[#ThisPtr]]
; CHECK-DAG: %[[#FunPtrTy:]] = OpTypePointer CodeSectionINTEL %[[#FunTy]]
; CHECK-DAG: %[[#]] = OpConstantFunctionPointerINTEL %[[#FunPtrTy]] %[[#Dtor1:]]
; CHECK-DAG: %[[#]] = OpConstantFunctionPointerINTEL %[[#FunPtrTy]] %[[#Dtor2:]]
; CHECK: %[[#Dtor1]] = OpFunction %[[#]] None %[[#FunTy]]
; CHECK: %[[#Dtor2]] = OpFunction %[[#]] None %[[#FunTy]]

@vt = addrspace(1) global { [2 x ptr addrspace(1)] } { [2 x ptr addrspace(1)] [
  ptr addrspace(1) addrspacecast (ptr addrspace(9) @dtor1 to ptr addrspace(1)),
  ptr addrspace(1) addrspacecast (ptr addrspace(9) @dtor2 to ptr addrspace(1))] }

define spir_func void @dtor1(ptr addrspace(4) %this) addrspace(9) {
  store ptr addrspace(1) @vt, ptr addrspace(4) %this
  ret void
}

define spir_func void @dtor2(ptr addrspace(4) %this) addrspace(9) {
  store ptr addrspace(1) @vt, ptr addrspace(4) %this
  ret void
}
