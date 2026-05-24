; Check that we don't end up with duplicated array types in TypeMap.
; No FileCheck needed, we only want to check the absence of errors.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o -
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#]] = OpTypeArray %[[#]] %[[#]]
; CHECK-NOT: OpTypeArray

%duplicate = type { [2 x ptr addrspace(4)] }

define spir_kernel void @foo() {
entry:
  alloca [2 x ptr addrspace(4)], align 8
  alloca %duplicate, align 8
  ret void
}
