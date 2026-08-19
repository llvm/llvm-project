; RUN: opt -mtriple=amdgcn-amd-amdhsa -inline-call-penalty=0 -inline-threshold=100 -passes=inline -S < %s | FileCheck %s --check-prefixes=CHECK,AMDGPU
; RUN: opt -inline-call-penalty=0 -inline-threshold=100 -passes=inline -S < %s | FileCheck %s --check-prefixes=CHECK,GENERIC

@g = external global i32

define void @callee() {
  store volatile i32 1, ptr @g
  ret void
}

define void @caller_unreachable() {
; CHECK-LABEL: @caller_unreachable
; AMDGPU:      store volatile i32 1, ptr @g
; AMDGPU-NOT:  call void @callee
; GENERIC:     call void @callee
  call void @callee()
  unreachable
}

define void @caller_reachable() {
; CHECK-LABEL: @caller_reachable
; CHECK-NOT:   call void @callee
  call void @callee()
  ret void
}

declare void @llvm.trap()
declare i32 @llvm.amdgcn.workitem.id.x()

define internal void @emit(ptr addrspace(1) %p) {
  store volatile i32 1, ptr addrspace(1) %p
  store volatile i32 2, ptr addrspace(1) %p
  store volatile i32 3, ptr addrspace(1) %p
  ret void
}

define internal void @report_and_die(ptr addrspace(1) %p) noreturn {
  call void @emit(ptr addrspace(1) %p)
  call void @llvm.trap()
  unreachable
}

define amdgpu_kernel void @kernel(ptr addrspace(1) %p, i32 %n) {
; CHECK-LABEL: @kernel
; AMDGPU-NOT:  call void @report_and_die
; AMDGPU-NOT:  call void @emit
; AMDGPU:      store volatile i32 1, ptr addrspace(1)
; GENERIC:     call void @report_and_die
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %c = icmp slt i32 %id, %n
  br i1 %c, label %ok, label %fail
fail:
  call void @report_and_die(ptr addrspace(1) %p)
  unreachable
ok:
  ret void
}
