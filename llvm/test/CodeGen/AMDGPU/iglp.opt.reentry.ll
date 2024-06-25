; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -O3 < %s | FileCheck %s

; Test should not result in build failure
; CHECK-LABEL: shouldNotReApply

define amdgpu_kernel void @shouldNotReApply() {
entry:
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  store <4 x i32> zeroinitializer, ptr addrspace(3) null, align 2147483648
  tail call void @llvm.amdgcn.sched.group.barrier(i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  store i32 0, ptr addrspace(5) null, align 2147483648
  tail call void @llvm.amdgcn.sched.group.barrier(i32 0, i32 0, i32 0)
  ret void
}
