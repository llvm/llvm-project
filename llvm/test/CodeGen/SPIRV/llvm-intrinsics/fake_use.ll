; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Addresses
; CHECK-DAG: OpName %[[#]] "foo"

declare void @llvm.fake.use(...)

define spir_kernel void @foo(ptr addrspace(1) %a) {
entry:
  call void (...) @llvm.fake.use(ptr addrspace(1) %a)
  ret void
}
