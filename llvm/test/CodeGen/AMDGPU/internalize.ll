; RUN: opt -O1 -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck %s
; CHECK-NOT: unused
; CHECK-NOT: foo_used
; CHECK: gvar_used
; CHECK: main_kernel

@gvar_unused = addrspace(1) global i32 undef, align 4
@gvar_used = addrspace(1) global i32 undef, align 4

; Function Attrs: alwaysinline nounwind
define void @foo_unused(i32 addrspace(1)* %out) local_unnamed_addr #1 {
entry:
  store i32 1, i32 addrspace(1)* %out
  ret void
}

; Function Attrs: alwaysinline nounwind
define void @foo_used(i32 addrspace(1)* %out, i32 %tid) local_unnamed_addr #1 {
entry:
  store i32 %tid, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @main_kernel() {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  tail call void @foo_used(i32 addrspace(1)* @gvar_used, i32 %tid) nounwind
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }

attributes #1 = { alwaysinline nounwind }
