; RUN: llc -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Verify that llvm.compiler.used is not lowered.
; CHECK: OpName %{{[0-9]+}} "unused"
; CHECK-NOT: OpName %{{[0-9]+}} "llvm.compiler.used"

; Check that the type of llvm.compiler.used is not emitted too.
; CHECK-NOT: OpTypeArray

@unused = private addrspace(3) global i32 0
@llvm.compiler.used = appending addrspace(2) global [1 x ptr addrspace (4)] [ptr addrspace(4) addrspacecast (ptr addrspace(3) @unused to ptr addrspace(4))]

define spir_func void @foo() {
entry:
  ret void
}
