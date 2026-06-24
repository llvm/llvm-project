; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpConvertPtrToU

define spir_kernel void @test_ptrtoaddr(ptr addrspace(1) %p, ptr addrspace(1) %res) {
entry:
  %addr = ptrtoaddr ptr addrspace(1) %p to i64
  store i64 %addr, ptr addrspace(1) %res
  ret void
}
