; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: "foo"
define spir_kernel void @foo(ptr addrspace(1) %a) {
entry:
  %a.addr = alloca ptr addrspace(1), align 4
  store ptr addrspace(1) %a, ptr %a.addr, align 4
  %0 = load ptr addrspace(1), ptr %a.addr, align 4
; CHECK: OpStore %[[#]] %[[#]] Aligned 4
  store i32 0, ptr addrspace(1) %0, align 4
  ret void
}
