; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

declare ptr addrspace(11) @llvm.spv.ptrcast(ptr addrspace(11), metadata, i32)

define void @test_atomic_rmw_ptrcast(ptr addrspace(11) %ptr) {
; CHECK: OpAtomicAnd
entry:
  %casted = call ptr addrspace(11) @llvm.spv.ptrcast(ptr addrspace(11) %ptr, metadata !"i32", i32 11)
  %atomic = atomicrmw and ptr addrspace(11) %casted, i32 1 syncscope("device") monotonic, align 4
  ret void
}

define void @test_atomic_rmw_ptrcast_add(ptr addrspace(11) %ptr) {
; CHECK: OpAtomicIAdd
entry:
  %casted = call ptr addrspace(11) @llvm.spv.ptrcast(ptr addrspace(11) %ptr, metadata !"i32", i32 11)
  %atomic = atomicrmw add ptr addrspace(11) %casted, i32 1 syncscope("device") monotonic, align 4
  ret void
}
