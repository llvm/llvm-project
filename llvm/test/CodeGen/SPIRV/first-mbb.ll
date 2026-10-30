; Test to ensure that the MachineIR Builder is always in a valid state, even when the first action is type or constant creation.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Addresses
; CHECK-DAG: OpCapability Kernel
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]]

declare void @llvm.memcpy.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)

define spir_func void @f(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %n) {
  call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %n, i1 false)
  ret void
}
