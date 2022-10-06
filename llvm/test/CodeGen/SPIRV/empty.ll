; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpCapability Addresses
; CHECK: "foo"
define spir_kernel void @foo(i32 addrspace(1)* %a) {
entry:
  %a.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %a, i32 addrspace(1)** %a.addr, align 4
  ret void
}
