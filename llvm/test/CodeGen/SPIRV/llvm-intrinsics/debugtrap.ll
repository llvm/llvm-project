; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpNop
; CHECK-NEXT: OpReturn

declare void @llvm.debugtrap()

define spir_kernel void @foo(ptr addrspace(1) %a){
entry:
  %a.addr = alloca ptr addrspace(1), align 4
  store ptr addrspace(1) %a, ptr %a.addr, align 4
  call void @llvm.debugtrap()
  ret void
}
