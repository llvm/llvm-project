; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-hoist-guarded-loads < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; An all-pairs loop skips its own element: the load of pos[j] runs when
; j != i, and pos[i] was loaded before the loop, so the load runs at the top
; of every iteration.
; CHECK-LABEL: define agx_kernel void @pairs(
; CHECK:       loop:
; CHECK:         %j = phi i32
; CHECK:         load float, ptr addrspace(1)
; CHECK:         br i1 %self, label %next, label %body
; CHECK:       body:
; CHECK-NOT:     load
; CHECK:       next:
define agx_kernel void @pairs(ptr addrspace(1) %pos, ptr addrspace(1) %out, i32 %i, i32 %n) {
entry:
  %ie = sext i32 %i to i64
  %pi = getelementptr float, ptr addrspace(1) %pos, i64 %ie
  %xi = load float, ptr addrspace(1) %pi, align 4
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j.next, %next ]
  %acc = phi float [ 0.0, %entry ], [ %acc2, %next ]
  %self = icmp eq i32 %j, %i
  br i1 %self, label %next, label %body
body:
  %je = sext i32 %j to i64
  %pj = getelementptr float, ptr addrspace(1) %pos, i64 %je
  %xj = load float, ptr addrspace(1) %pj, align 4
  %d = fsub float %xi, %xj
  %acc1 = fadd float %acc, %d
  br label %next
next:
  %acc2 = phi float [ %acc, %loop ], [ %acc1, %body ]
  %j.next = add nsw i32 %j, 1
  %done = icmp eq i32 %j.next, %n
  br i1 %done, label %exit, label %loop
exit:
  store float %acc2, ptr addrspace(1) %out
  ret void
}

; A loop that writes memory keeps its load where it is.
; CHECK-LABEL: define agx_kernel void @writes(
; CHECK:       body:
; CHECK:         load float, ptr addrspace(1)
define agx_kernel void @writes(ptr addrspace(1) %pos, ptr addrspace(1) %out, i32 %i, i32 %n) {
entry:
  %ie = sext i32 %i to i64
  %pi = getelementptr float, ptr addrspace(1) %pos, i64 %ie
  %xi = load float, ptr addrspace(1) %pi, align 4
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j.next, %next ]
  %self = icmp eq i32 %j, %i
  br i1 %self, label %next, label %body
body:
  %je = sext i32 %j to i64
  %pj = getelementptr float, ptr addrspace(1) %pos, i64 %je
  %xj = load float, ptr addrspace(1) %pj, align 4
  %d = fsub float %xi, %xj
  %po = getelementptr float, ptr addrspace(1) %out, i64 %je
  store float %d, ptr addrspace(1) %po
  br label %next
next:
  %j.next = add nsw i32 %j, 1
  %done = icmp eq i32 %j.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; Without the element of the skipped index loaded before the loop, the load
; stays (pos[n] may not exist).
; CHECK-LABEL: define agx_kernel void @unloaded(
; CHECK:       body:
; CHECK:         load float, ptr addrspace(1)
define agx_kernel void @unloaded(ptr addrspace(1) %pos, ptr addrspace(1) %out, i32 %i, i32 %n) {
entry:
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j.next, %next ]
  %acc = phi float [ 0.0, %entry ], [ %acc2, %next ]
  %skip = icmp eq i32 %j, %i
  br i1 %skip, label %next, label %body
body:
  %je = sext i32 %j to i64
  %pj = getelementptr float, ptr addrspace(1) %pos, i64 %je
  %xj = load float, ptr addrspace(1) %pj, align 4
  %acc1 = fadd float %acc, %xj
  br label %next
next:
  %acc2 = phi float [ %acc, %loop ], [ %acc1, %body ]
  %j.next = add nsw i32 %j, 1
  %done = icmp eq i32 %j.next, %n
  br i1 %done, label %exit, label %loop
exit:
  store float %acc2, ptr addrspace(1) %out
  ret void
}
