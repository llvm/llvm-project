; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

; These tests have identical control flow graphs with slight changes
; that affect cycle-info. There is a minor functional difference in
; the branch conditions; but that is not relevant to the tests.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; The cycle has a header (T) that does not dominate the join, hence
;; the entire cycle is reported as converged.
;;
;; CHECK-LABEL: UniformityInfo for function 't_header':
;; CHECK: CYCLES ASSSUMED DIVERGENT:
;; CHECK:   depth=1: entries(T P) S Q R

define amdgpu_kernel void @t_header(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 %a.div = add i32 %tid, %a
 br i1 %cond.uni, label %P, label %T

P:
; CHECK:   DIVERGENT:   %pp.phi =
  %pp.phi  = phi i32 [ %a, %entry], [ %b, %T ]
  %pp = add i32 %b, 1
  br i1 %cond.uni, label %R, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
; CHECK:   DIVERGENT:   %s.phi =
; CHECK:   DIVERGENT:   %ss =
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %pp.phi, 1
  br i1 %cond.uni, label %exit, label %T

T:
; CHECK:   DIVERGENT:   %tt.phi =
  %tt.phi = phi i32 [ %ss, %S ], [ %a, %entry ]
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; The cycle has a header (P) that dominates the join, hence
;; the cycle is reported as converged.
;;
;; CHECK-LABEL: UniformityInfo for function 'p_header':
;; CHECK-NOT: CYCLES ASSSUMED DIVERGENT:

define amdgpu_kernel void @p_header(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 br i1 %cond.uni, label %T, label %P

P:
; CHECK-NOT:   DIVERGENT:   %pp.phi = phi i32
  %pp.phi  = phi i32 [ %a, %entry], [ %b, %T ]
  %pp = add i32 %b, 1
  br i1 %cond.uni, label %R, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
; CHECK:   DIVERGENT:   %s.phi =
; CHECK-NOT:   DIVERGENT:   %ss = add i32
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %pp.phi, 1
  br i1 %cond.uni, label %exit, label %T

T:
; CHECK-NIT:   DIVERGENT:   %tt.phi = phi i32
  %tt.phi = phi i32 [ %ss, %S ], [ %a, %entry ]
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
