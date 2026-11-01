; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -unroll-partial-threshold=60 -pass-remarks=loop-unroll-and-jam -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -unroll-partial-threshold=60 -unroll-and-jam-max-reuse-count=2 -pass-remarks=loop-unroll-and-jam -disable-output < %s 2>&1 | FileCheck %s --check-prefix=MAX2
; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -unroll-partial-threshold=60 -unroll-and-jam-min-reuse-percent=50 -pass-remarks=loop-unroll-and-jam -disable-output < %s 2>&1 | FileCheck %s --check-prefix=PCT50

; The unroll count comes from the distance between accesses that become
; redundant in the jammed loop.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; c[i][j] = a[i][j] + a[i+1][j]
; CHECK: remark: <unknown>:0:0: unroll and jammed loop by a factor of 2 with run-time trip count
; MAX2: remark: <unknown>:0:0: unroll and jammed loop by a factor of 2 with run-time trip count
; One of the three accesses goes away, which isn't enough for 50%.
define void @distance1(i64 %n, i64 %m, ptr noalias %c, ptr noalias %a) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %i.next = add nuw nsw i64 %i, 1
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %p0 = getelementptr inbounds [1000 x double], ptr %a, i64 %i, i64 %j
  %p1 = getelementptr inbounds [1000 x double], ptr %a, i64 %i.next, i64 %j
  %v0 = load double, ptr %p0
  %v1 = load double, ptr %p1
  %sum = fadd double %v0, %v1
  %q = getelementptr inbounds [1000 x double], ptr %c, i64 %i, i64 %j
  store double %sum, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; c[i][j] = a[i][j] + a[i+3][j]
; CHECK: remark: <unknown>:0:0: unroll and jammed loop by a factor of 4 with run-time trip count
define void @distance3(i64 %n, i64 %m, ptr noalias %c, ptr noalias %a) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %i.3 = add nuw nsw i64 %i, 3
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %p0 = getelementptr inbounds [1000 x double], ptr %a, i64 %i, i64 %j
  %p1 = getelementptr inbounds [1000 x double], ptr %a, i64 %i.3, i64 %j
  %v0 = load double, ptr %p0
  %v1 = load double, ptr %p1
  %sum = fadd double %v0, %v1
  %q = getelementptr inbounds [1000 x double], ptr %c, i64 %i, i64 %j
  store double %sum, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; c[i][j] = a[i][j] + a[i+4][j]: the two loads never share a group of 4 copies.
define void @distance_too_large(i64 %n, i64 %m, ptr noalias %c, ptr noalias %a) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %i.4 = add nuw nsw i64 %i, 4
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %p0 = getelementptr inbounds [1000 x double], ptr %a, i64 %i, i64 %j
  %p1 = getelementptr inbounds [1000 x double], ptr %a, i64 %i.4, i64 %j
  %v0 = load double, ptr %p0
  %v1 = load double, ptr %p1
  %sum = fadd double %v0, %v1
  %q = getelementptr inbounds [1000 x double], ptr %c, i64 %i, i64 %j
  store double %sum, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; c[i][j] = a[i][j] + b[i][j]: nothing is accessed twice.
define void @no_reuse(i64 %n, i64 %m, ptr noalias %c, ptr noalias %a, ptr noalias %b) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %p0 = getelementptr inbounds [1000 x double], ptr %a, i64 %i, i64 %j
  %p1 = getelementptr inbounds [1000 x double], ptr %b, i64 %i, i64 %j
  %v0 = load double, ptr %p0
  %v1 = load double, ptr %p1
  %sum = fadd double %v0, %v1
  %q = getelementptr inbounds [1000 x double], ptr %c, i64 %i, i64 %j
  store double %sum, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; c[i*m+j] = a[i*m+j] + a[(i+1)*m+j], with a stride only known at run time.
; CHECK: remark: <unknown>:0:0: unroll and jammed loop by a factor of 2 with run-time trip count
; MAX2: remark: <unknown>:0:0: unroll and jammed loop by a factor of 2 with run-time trip count
define void @runtime_stride(i64 %n, i64 %m, ptr noalias %c, ptr noalias %a) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %base = phi i64 [ 0, %entry ], [ %base.next, %outer.latch ]
  %base.next = add nsw i64 %base, %m
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx0 = add nsw i64 %base, %j
  %idx1 = add nsw i64 %base.next, %j
  %p0 = getelementptr inbounds double, ptr %a, i64 %idx0
  %p1 = getelementptr inbounds double, ptr %a, i64 %idx1
  %v0 = load double, ptr %p0
  %v1 = load double, ptr %p1
  %sum = fadd double %v0, %v1
  %q = getelementptr inbounds double, ptr %c, i64 %idx0
  store double %sum, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; c[i][j] = b[j]: the load is shared by all copies, so the size limits decide.
; CHECK: remark: <unknown>:0:0: unroll and jammed loop by a factor of {{[0-9]+}}
; MAX2: remark: <unknown>:0:0: unroll and jammed loop by a factor of {{[0-9]+}}
; PCT50: remark: <unknown>:0:0: unroll and jammed loop by a factor of {{[0-9]+}}
define void @invariant_load(i64 %n, i64 %m, ptr noalias %c, ptr noalias %b) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %p = getelementptr inbounds double, ptr %b, i64 %j
  %v = load double, ptr %p
  %q = getelementptr inbounds [1000 x double], ptr %c, i64 %i, i64 %j
  store double %v, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; CHECK-NOT: remark:
; MAX2-NOT: remark:
; PCT50-NOT: remark:
