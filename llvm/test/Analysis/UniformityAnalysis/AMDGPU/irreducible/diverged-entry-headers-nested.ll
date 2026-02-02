; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

; These tests have identical control flow graphs with slight changes
; that affect cycle-info. There is a minor functional difference in
; the branch conditions; but that is not relevant to the tests.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; The inner cycle has a header (P) that dominates the join, hence
;; both cycles are reported as converged.
;;
;; CHECK-LABEL: UniformityInfo for function 'headers_b_p':
;; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
;; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:

define amdgpu_kernel void @headers_b_p(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 %a.div = add i32 %tid, %a
 br i1 %cond.uni, label %B, label %A

A:
 br label %B

B:
 br i1 %cond.uni, label %C, label %D

C:
 br i1 %cond.uni, label %T, label %P

P:
  %pp.phi  = phi i32 [ %a, %C], [ %b, %T ]
  %pp = add i32 %b, 1
  br i1 %cond.uni, label %R, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %pp.phi, 1
  br i1 %cond.uni, label %D, label %T

D:
  br i1 %cond.uni, label %exit, label %A

T:
  %tt.phi = phi i32 [ %ss, %S ], [ %a, %C ]
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Same as previous, but the outer cycle has a different header (A).
;; The inner cycle has a header (P) that dominates the join, hence
;; both cycles are reported as converged.
;;
;; CHECK-LABEL: UniformityInfo for function 'headers_a_p':
;; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
;; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:

define amdgpu_kernel void @headers_a_p(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 %a.div = add i32 %tid, %a
 br i1 %cond.uni, label %B, label %A

A:
 br label %B

B:
 br i1 %cond.uni, label %C, label %D

C:
 br i1 %cond.uni, label %T, label %P

P:
  %pp.phi  = phi i32 [ %a, %C], [ %b, %T ]
  %pp = add i32 %b, 1
  br i1 %cond.uni, label %R, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %pp.phi, 1
  br i1 %cond.uni, label %D, label %T

D:
  br i1 %cond.uni, label %exit, label %A

T:
  %tt.phi = phi i32 [ %ss, %S ], [ %a, %C ]
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; The inner cycle has a header (T) that does not dominate the join.
;; The outer cycle has a header (B) that dominates the join. Hence
;; only the inner cycle is reported as diverged.
;;
;; CHECK-LABEL: UniformityInfo for function 'headers_b_t':
;; CHECK: CYCLES ASSUMED DIVERGENT:
;; CHECK:   depth=2: entries(T P) S Q R
;; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:

define amdgpu_kernel void @headers_b_t(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 %a.div = add i32 %tid, %a
 br i1 %cond.uni, label %A, label %B

A:
 br label %B

B:
 br i1 %cond.uni, label %C, label %D

C:
 br i1 %cond.uni, label %P, label %T

P:
  %pp.phi  = phi i32 [ %a, %C], [ %b, %T ]
  %pp = add i32 %b, 1
  br i1 %cond.uni, label %R, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %pp.phi, 1
  br i1 %cond.uni, label %D, label %T

D:
  br i1 %cond.uni, label %exit, label %A

T:
  %tt.phi = phi i32 [ %ss, %S ], [ %a, %C ]
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; The cycles have headers (A and T) that do not dominate the join.
;; Hence the outermost cycle is reported as diverged.
;;
;; CHECK-LABEL: UniformityInfo for function 'headers_a_t':
;; CHECK: CYCLES ASSUMED DIVERGENT:
;; CHECK:   depth=1: entries(A B) D T S Q P R C
;; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:

define amdgpu_kernel void @headers_a_t(i32 %a, i32 %b, i32 %c) {
entry:
 %cond.uni = icmp slt i32 %a, 0
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %cond.div = icmp slt i32 %tid, 0
 %a.div = add i32 %tid, %a
 br i1 %cond.uni, label %B, label %A

A:
 br label %B

B:
 br i1 %cond.uni, label %C, label %D

C:
 br i1 %cond.uni, label %P, label %T

P:
  %pp.phi  = phi i32 [ %a, %C], [ %b, %T ]
  %pp = add i32 %b, 1
  br i1 %cond.uni, label %R, label %Q

Q:
  %qq = add i32 %b, 1
  br i1 %cond.div, label %S, label %R

R:
  %rr = add i32 %b, 1
  br label %S

S:
  %s.phi = phi i32 [ %qq, %Q ], [ %rr, %R ]
  %ss = add i32 %pp.phi, 1
  br i1 %cond.uni, label %D, label %T

D:
  br i1 %cond.uni, label %exit, label %A

T:
  %tt.phi = phi i32 [ %ss, %S ], [ %a, %C ]
  %tt = add i32 %b, 1
  br label %P

exit:
  %ee = add i32 %b, 1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
