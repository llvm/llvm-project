; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; addrspacecast between untyped pointers emits OpPtrCastToGeneric and needs the
; GenericPointer capability.

; CHECK-DAG: OpCapability UntypedPointersKHR
; CHECK-DAG: OpCapability GenericPointer
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#CROSS:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#GEN:]] = OpTypeUntypedPointerKHR Generic

; CHECK: %[[#G:]] = OpFunctionParameter %[[#CROSS]]
; CHECK: %[[#CAST:]] = OpPtrCastToGeneric %[[#GEN]] %[[#G]]
; CHECK: OpStore %[[#CAST]]
define spir_kernel void @test(ptr addrspace(1) %g) {
  %gen = addrspacecast ptr addrspace(1) %g to ptr addrspace(4)
  store i32 3, ptr addrspace(4) %gen, align 4
  ret void
}
