; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define amdgpu_kernel void @kernel(ptr addrspace(1) %out, i32 %n) {
entry:
; CHECK: a kernel may not call a kernel
; CHECK-NEXT: ptr @kernel
  call void @kernel(ptr addrspace(1) %out, i32 %n)
  ret void
}
