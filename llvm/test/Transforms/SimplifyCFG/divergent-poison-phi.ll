; REQUIRES: amdgpu-registered-target && x86-registered-target
; RUN: opt < %s -mtriple=amdgcn -passes=simplifycfg -S | FileCheck %s -check-prefix=DIVERGENT
; RUN: opt < %s -mtriple=x86_64 -passes=simplifycfg -S | FileCheck %s -check-prefix=UNIFORM

; When poison on a PHI incoming edge flows to a convergent noundef call at a
; join point, do not treat that edge as unreachable. Scalar folding remains
; valid for the same pattern with a non-convergent callee.

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @shuffle(i32 noundef, i32 noundef) convergent

declare i32 @use_val(i32 noundef)

define amdgpu_kernel void @k_convergent(ptr addrspace(1) nocapture %counter, ptr addrspace(1) nocapture %out) #1 {
; DIVERGENT-LABEL: @k_convergent(
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %old = atomicrmw add ptr addrspace(1) %counter, i32 1 monotonic, align 4
  br label %if.end

if.end:
  %val = phi i32 [ %old, %if.then ], [ poison, %entry ]
  %shfl = tail call i32 @shuffle(i32 noundef %val, i32 noundef %tid) #2
  %idx = zext i32 %tid to i64
  %ptr = getelementptr i32, ptr addrspace(1) %out, i64 %idx
  store i32 %shfl, ptr addrspace(1) %ptr, align 4
  ret void
}

define void @k_scalar(i32 %tid, ptr nocapture %counter, ptr nocapture %out) {
; UNIFORM-LABEL: @k_scalar(
entry:
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %old = atomicrmw add ptr %counter, i32 1 monotonic, align 4
  br label %if.end

if.end:
  %val = phi i32 [ %old, %if.then ], [ poison, %entry ]
  %use = call i32 @use_val(i32 noundef %val)
  %idx = zext i32 %tid to i64
  %ptr = getelementptr i32, ptr %out, i64 %idx
  store i32 %use, ptr %ptr, align 4
  ret void
}

; DIVERGENT-NOT: call void @llvm.assume
; DIVERGENT: br i1 {{%.*}}, label %if.then, label %if.end
; DIVERGENT: atomicrmw add
; DIVERGENT: phi i32

; UNIFORM: call void @llvm.assume
; UNIFORM-NOT: br i1 {{%.*}}, label %if.then, label %if.end

attributes #0 = { nounwind readnone }
attributes #1 = { convergent "amdgpu-flat-work-group-size"="1,256" }
attributes #2 = { convergent }
