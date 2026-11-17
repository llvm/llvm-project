; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-uniform-intrinsic-combine -S < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 -S < %s | FileCheck %s --check-prefix=O3
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-enable-uniform-intrinsic-combine=0 -O3 -S < %s | FileCheck %s --check-prefix=COMBINE-OFF

declare i32 @llvm.amdgcn.readfirstlane.i32(i32)
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.ballot.i32(i1)
declare i1 @llvm.is.constant.i32(i32)
declare i32 @llvm.amdgcn.permlane.down.i32(i32, i32, i32)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)

; CHECK-LABEL: @all_predicates(
; CHECK:      store volatile i1 true
; CHECK-NEXT: store volatile i1 false
; CHECK-NEXT: store volatile i1 false
; CHECK-NEXT: store volatile i1 true
; CHECK-NEXT: store volatile i1 false
; CHECK-NEXT: store volatile i1 true
; CHECK-NEXT: store volatile i1 false
; CHECK-NEXT: store volatile i1 true
; CHECK-NEXT: store volatile i1 false
; CHECK-NEXT: store volatile i1 true
; CHECK-NOT:  icmp
define amdgpu_kernel void @all_predicates(ptr addrspace(1) %out, i32 %x) {
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %x)
  %a = icmp eq  i32 %rfl, %x
  %b = icmp ne  i32 %rfl, %x
  %c = icmp sgt i32 %rfl, %x
  %d = icmp sge i32 %rfl, %x
  %e = icmp slt i32 %rfl, %x
  %f = icmp sle i32 %rfl, %x
  %g = icmp ugt i32 %rfl, %x
  %h = icmp uge i32 %rfl, %x
  %i = icmp ult i32 %rfl, %x
  %j = icmp ule i32 %rfl, %x
  store volatile i1 %a, ptr addrspace(1) %out
  store volatile i1 %b, ptr addrspace(1) %out
  store volatile i1 %c, ptr addrspace(1) %out
  store volatile i1 %d, ptr addrspace(1) %out
  store volatile i1 %e, ptr addrspace(1) %out
  store volatile i1 %f, ptr addrspace(1) %out
  store volatile i1 %g, ptr addrspace(1) %out
  store volatile i1 %h, ptr addrspace(1) %out
  store volatile i1 %i, ptr addrspace(1) %out
  store volatile i1 %j, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @commuted_match(
; CHECK:     store i1 false
; CHECK-NOT: icmp
define amdgpu_kernel void @commuted_match(ptr addrspace(1) %out, i32 %x) {
  %r1 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %x)
  %r2 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %r1)
  %cmp = icmp slt i32 %r1, %r2
  store i1 %cmp, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @loop_phi_uniform_step(
; CHECK:     store i32
; CHECK-NOT: icmp eq
define amdgpu_kernel void @loop_phi_uniform_step(ptr addrspace(1) %out, i32 %step, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %sum, %loop ]
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i)
  %cmp = icmp eq i32 %rfl, %i
  %sel = select i1 %cmp, i32 %rfl, i32 0
  %sum = add i32 %acc, %sel
  %next = add i32 %i, %step
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @readfirstlane_multi_use(
; CHECK:     call i32 @llvm.amdgcn.readfirstlane
; CHECK:     select i1 true
; CHECK-NOT: icmp
define amdgpu_kernel void @readfirstlane_multi_use(ptr addrspace(1) %out, i32 %x) {
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %x)
  %cmp = icmp eq i32 %rfl, %x
  %sel = select i1 %cmp, i32 %rfl, i32 0
  store i32 %sel, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @split_uses_across_divergent_exit(
; CHECK:     store i1 true
; CHECK-NOT: icmp eq
define amdgpu_kernel void @split_uses_across_divergent_exit(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i)
  %inc = add i32 %i, 1
  %done = icmp sgt i32 %inc, %tid
  br i1 %done, label %exit, label %loop
exit:
  %cmp = icmp eq i32 %rfl, %i
  store i1 %cmp, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @divergent_sources(
; CHECK: icmp eq i32 %rfl1, %tid
; CHECK: icmp eq i32 %rfl2, %add
; CHECK: icmp slt i32 %rfl3, %tid
define amdgpu_kernel void @divergent_sources(ptr addrspace(1) %out, i32 %x) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %rfl1 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %tid)
  %a = icmp eq i32 %rfl1, %tid
  %add = add i32 %x, %tid
  %rfl2 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %add)
  %b = icmp eq i32 %rfl2, %add
  %rfl3 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %tid)
  %c = icmp slt i32 %rfl3, %tid
  store volatile i1 %a, ptr addrspace(1) %out
  store volatile i1 %b, ptr addrspace(1) %out
  store volatile i1 %c, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @temporal_divergence(
; CHECK: icmp eq
define amdgpu_kernel void @temporal_divergence(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %i, 1
  %done = icmp sgt i32 %inc, %tid
  br i1 %done, label %exit, label %loop
exit:
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i)
  %cmp = icmp eq i32 %rfl, %i
  store i1 %cmp, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @different_ssa_value(
; CHECK: icmp eq
define amdgpu_kernel void @different_ssa_value(ptr addrspace(1) %out, i32 %x0, i32 %n) {
entry:
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %x0)
  br label %loop
loop:
  %x = phi i32 [ %x0, %entry ], [ %next, %loop ]
  %cmp = icmp eq i32 %rfl, %x
  %next = shl i32 %x, 1
  %d = icmp sge i32 %next, %n
  br i1 %d, label %exit, label %loop
exit:
  store i1 %cmp, ptr addrspace(1) %out
  ret void
}

; O3-LABEL: @guarded_shuffle_loop(
; O3-NOT: ds.bpermute
; O3:     call i32 @llvm.amdgcn.permlane.down.i32
; COMBINE-OFF-LABEL: @guarded_shuffle_loop(
; COMBINE-OFF-NOT: permlane.down
; COMBINE-OFF:     call i32 @llvm.amdgcn.ds.bpermute
define amdgpu_kernel void @guarded_shuffle_loop(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %v = load i32, ptr addrspace(1) %gep
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %next, %latch ]
  %acc = phi i32 [ %v, %entry ], [ %sum, %latch ]
  %rfl = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %i)
  %ne = icmp ne i32 %rfl, %i
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 %ne)
  %isc = call i1 @llvm.is.constant.i32(i32 %ballot)
  %z = icmp eq i32 %ballot, 0
  %uniform = and i1 %isc, %z
  br i1 %uniform, label %fast, label %slow
fast:
  %a = call i32 @llvm.amdgcn.permlane.down.i32(i32 %acc, i32 %rfl, i32 32)
  br label %latch
slow:
  %lane = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %idx = add i32 %lane, %i
  %byte = shl i32 %idx, 2
  %b = call i32 @llvm.amdgcn.ds.bpermute(i32 %byte, i32 %acc)
  br label %latch
latch:
  %sel = phi i32 [ %a, %fast ], [ %b, %slow ]
  %sum = add i32 %sel, %acc
  %next = shl i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @uniform_waterfall(
; CHECK:      readfirstlane.i32(i32 %mymask)
; CHECK-NEXT: br i1 true, label %{{.*}}, label %{{.*}}
; O3-LABEL: @uniform_waterfall(
; O3:      store i32 5, ptr addrspace(1) %out
; O3-NEXT: ret void
define amdgpu_kernel void @uniform_waterfall(ptr addrspace(1) %out, i32 %mymask) {
entry:
  br label %while
while:
  %done = phi i1 [ false, %entry ], [ %new_done, %tail ]
  %not_done = xor i1 %done, true
  %ballot = tail call i32 @llvm.amdgcn.ballot.i32(i1 %not_done)
  %is_done = icmp eq i32 %ballot, 0
  br i1 %is_done, label %exit, label %if
if:
  %first_active_id = tail call noundef i32 @llvm.amdgcn.readfirstlane.i32(i32 %mymask)
  %is_first_active_id = icmp eq i32 %mymask, %first_active_id
  br i1 %is_first_active_id, label %work, label %tail
work:
  store i32 5, ptr addrspace(1) %out
  br label %tail
tail:
  %new_done = phi i1 [ true, %work ], [ false, %if ]
  br label %while
exit:
  ret void
}
