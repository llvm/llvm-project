; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s 2>&1 | FileCheck %s

define amdgpu_kernel void @entry_fn() {
; CHECK-NOT: LLVM ERROR: invalid call to entry function
entry:
  call void @entry_fn()
  ret void
}
