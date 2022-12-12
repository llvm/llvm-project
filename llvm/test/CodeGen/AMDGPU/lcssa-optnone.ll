; RUN: llc -march=amdgcn -O0 -o - %s | FileCheck %s

; CHECK-LABEL: non_uniform_loop
; CHECK: s_endpgm
define amdgpu_kernel void @non_uniform_loop(ptr addrspace(1) %array) {
entry:
  %w = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %for.cond

for.cond:
  %i = phi i32 [0, %entry], [%i.next, %for.inc]
  %cmp = icmp ult i32 %i, %w
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %i.next = add i32 %i, 1
  br label %for.cond

for.end:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
