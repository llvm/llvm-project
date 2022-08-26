; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: %{{[0-9]+}} = OpExtInstImport "OpenCL.std"
; CHECK: %{{[0-9]+}} = OpTypeInt 32 0

define spir_kernel void @foo(i32 addrspace(1)* %a) {
entry:
  %a.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %a, i32 addrspace(1)** %a.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %a.addr, align 4
  store i32 0, i32 addrspace(1)* %0, align 4
  ret void
}
