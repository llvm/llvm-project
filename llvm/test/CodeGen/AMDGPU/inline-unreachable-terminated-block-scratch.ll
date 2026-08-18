; RUN: opt -passes=inline -inline-call-penalty=0 -mtriple=amdgpu9.50-amd-amdhsa < %s | llc -mtriple=amdgpu9.50-amd-amdhsa | FileCheck %s

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

; CHECK-LABEL: {{^}}kernel:
; CHECK: .amdhsa_private_segment_fixed_size 0
define amdgpu_kernel void @kernel(ptr addrspace(1) %p, i32 %n) {
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
