; RUN: llc -mtriple=amdgcn -mcpu=gfx900 - < %s | FileCheck %s

@local = addrspace(3) global i32 undef

define amdgpu_kernel void @reducible() {
; CHECK-LABEL: reducible:
; CHECK-NOT: dpp
entry:
  %x = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %gep = getelementptr i32, ptr addrspace(3) @local, i32 %i
  %cond = icmp ult i32 %i, %x
  %i1 = add i32 %i, 1
  br i1 %cond, label %loop, label %exit
exit:
  %old = atomicrmw add ptr addrspace(3) %gep, i32 %x acq_rel
  ret void
}

define amdgpu_kernel void @def_in_nested_cycle() {
; CHECK-LABEL: def_in_nested_cycle:
; CHECK-NOT: dpp
entry:
  %x = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ 0, %innerloop ], [ %i1, %loop ]
  %cond = icmp ult i32 %i, %x
  %i1 = add i32 %i, 1
  br i1 %cond, label %innerloop, label %loop
innerloop:
  %i.inner = phi i32 [ 0, %loop ], [ %i1.inner, %innerloop ]
  %gep = getelementptr i32, ptr addrspace(3) @local, i32 %i
  %i1.inner = add i32 %i, 1
  %cond.inner = icmp ult i32 %i, %x
  br i1 %cond, label %innerloop, label %loop
exit:
  %old = atomicrmw add ptr addrspace(3) %gep, i32 %x acq_rel
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
