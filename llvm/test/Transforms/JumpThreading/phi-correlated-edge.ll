; RUN: opt -S -passes=jump-threading,simplifycfg < %s | FileCheck %s

declare ptr @g()
declare i64 @h()
declare void @side()

define void @correlated_phi(i32 %n) {
; CHECK-LABEL: define void @correlated_phi(
entry:
  %p = call ptr @g()
  %isnull = icmp eq ptr %p, null
  br label %preheader

preheader:
  %iv = phi i32 [ 0, %entry ], [ %n, %crit_edge ]
  %flag = phi i1 [ %isnull, %entry ], [ true, %crit_edge ]
  br label %loop

issue_check:
  br i1 %flag, label %ret, label %issue_dead

crit_edge:
; CHECK: crit_edge:
; CHECK-NEXT:  %exit = icmp eq i32 %iv, 1
; CHECK-NEXT:  br i1 %exit, label %ret, label %preheader
  %exit = icmp eq i32 %iv, 1
  br i1 %exit, label %issue_check, label %preheader

loop:
  %size = call i64 @h()
  %empty = icmp eq i64 %size, 0
  br i1 %empty, label %crit_edge, label %loop

ret:
  ret void

issue_dead:
  %extra = call i64 @h()
  br label %ret
}

; CHECK-NOT: issue_dead:
; CHECK-NOT: %flag = phi

define void @correlated_phi_nonconstant(i1 %c, i1 %d) {
; CHECK-LABEL: define void @correlated_phi_nonconstant(
entry:
  br label %preheader

preheader:
  %iv = phi i32 [ 0, %entry ], [ 1, %backedge ]
  %flag = phi i1 [ %c, %entry ], [ %d, %backedge ]
  br label %backedge

backedge:
  %exit = icmp eq i32 %iv, 1
  br i1 %exit, label %neg_check, label %preheader

neg_check:
; CHECK: neg_check:
; CHECK-NEXT:  br i1 %flag, label %neg_ret, label %neg_dead
  br i1 %flag, label %neg_ret, label %neg_dead

neg_dead:
; CHECK: neg_dead:
; CHECK-NEXT:  call void @side()
  call void @side()
  br label %neg_ret

neg_ret:
  ret void
}
