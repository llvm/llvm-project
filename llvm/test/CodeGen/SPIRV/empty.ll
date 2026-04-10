; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Addresses
; CHECK: "foo"
define spir_kernel void @foo(ptr addrspace(1) %a) {
entry:
  %a.addr = alloca ptr addrspace(1), align 4
  store ptr addrspace(1) %a, ptr %a.addr, align 4
  ret void
}
