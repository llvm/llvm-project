; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_kernel void @kernel(ptr addrspace(1) %out, i32 %n) {
entry:
; CHECK: calling convention does not permit calls
; CHECK-NEXT: call void @kernel(ptr addrspace(1) %out, i32 %n)
  call void @kernel(ptr addrspace(1) %out, i32 %n)
  ret void
}
