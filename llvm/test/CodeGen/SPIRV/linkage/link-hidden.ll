; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Unknown function in:
; CHECK-SAME: OpFunctionCall %{{[0-9]+}}:type, @bar

define hidden spir_kernel void @foo() addrspace(4) {
entry:
  call spir_func addrspace(4) void @bar()
  ret void
}

declare hidden spir_func void @bar() addrspace(4)
