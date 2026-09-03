; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Atomic load/store/rmw through an untyped pointer.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0

; CHECK: %[[#P:]] = OpFunctionParameter %[[#PTR]]
; CHECK: OpAtomicStore %[[#P]] %[[#]] %[[#]] %[[#]]
; CHECK: OpAtomicLoad %[[#I32]] %[[#P]] %[[#]] %[[#]]
; CHECK: OpAtomicIAdd %[[#I32]] %[[#P]] %[[#]] %[[#]] %[[#]]
define spir_kernel void @test(ptr addrspace(1) %p) {
  store atomic i32 7, ptr addrspace(1) %p seq_cst, align 4
  %v = load atomic i32, ptr addrspace(1) %p seq_cst, align 4
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 seq_cst
  ret void
}
