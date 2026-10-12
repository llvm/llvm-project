; Check that we don't end up with duplicated array types in TypeMap.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#]] = OpTypeArray %[[#]] %[[#]]
; CHECK-NOT: OpTypeArray

%duplicate = type { [2 x ptr addrspace(4)] }

define spir_kernel void @foo() {
entry:
  %a = alloca [2 x ptr addrspace(4)], align 8
  %b = alloca %duplicate, align 8
  store ptr addrspace(4) null, ptr %a, align 8
  store ptr addrspace(4) null, ptr %b, align 8
  ret void
}
