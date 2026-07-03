; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

;
;                         Entry
;                           |
;                           v
;                  -------->H---------
;                  |        |        |
;                  |        v        |
;                  |    --->T----    |
;                  |    |       |    |
;                  |    |       V    |
;                  S<---R       P <---
;                  ^    ^       |
;                  |    |  Div  |
;                  |    --- Q <--
;                  |        |
;                  |        v
;                  -------- U
;                           |
;                           v
;                          Exit
;
; The divergent branch is at Q that exits an irreducible cycle with
; entries T and P nested inside a reducible cycle with header H. R is
; assigned label R, which reaches P. S is a join node with label S. If
; this is propagated to P via H, then P is incorrectly recognized as a
; join, making the inner cycle divergent. P is always executed
; convergently -- either by threads that reconverged at header H, or
; by threads that are still executing the inner cycle. Thus, any PHI
; at P should not be marked divergent.

define amdgpu_kernel void @nested_irreducible(i32 %a, i32 %b, i32 %c) {
; CHECK=LABEL: UniformityInfo for function 'nested_irreducible':
; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK-DAG:   depth=2: entries(P T) R Q
; CHECK-DAG:   depth=1: entries(H) S P T R Q U
entry:
  %cond.uni = icmp slt i32 %a, 0
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond.div = icmp slt i32 %tid, 0
  br label %H

H:
 br i1 %cond.uni, label %T, label %P

P:
; CHECK-LABEL: BLOCK P
; CHECK-NOT:   DIVERGENT:   %pp.phi =
; CHECK-NOT: DIVERGENT:      %pp =
  %pp.phi  = phi i32 [ %a, %H], [ %b, %T ]
  %pp = add i32 %b, 1
  br label %Q

Q:
; CHECK-LABEL: BLOCK Q
; CHECK-NOT: DIVERGENT:   %qq =
; CHECK-NOT:   DIVERGENT:   %qq.uni =
  %qq = add i32 %b, 1
  %qq.uni = add i32 %pp.phi, 1
  br i1 %cond.div, label %R, label %U

R:
  br i1 %cond.uni, label %S, label %T

T:
; CHECK-LABEL: BLOCK T
; CHECK-NOT:   DIVERGENT:   %tt.phi =
; CHECK-NOT: DIVERGENT:     %tt =
  %tt.phi = phi i32 [ %qq, %R ], [ %a, %H ]
  %tt = add i32 %b, 1
  br label %P

S:
; CHECK-LABEL: BLOCK S
; CHECK:   DIVERGENT:   %ss.phi =
; CHECK-NOT: DIVERGENT:     %ss =
  %ss.phi = phi i32 [ %qq.uni, %U ], [ %a, %R ]
  %ss = add i32 %b, 1
  br label %H

U:
  br i1 %cond.uni, label %S, label %exit

exit:
; CHECK: DIVERGENT:     %ee.div =
; CHECK-NOT: DIVERGENT:     %ee =
  %ee.div =  add i32 %qq.uni, 1
  %ee = add i32 %b, 1
  ret void
}

;
;                         Entry
;                           |
;                           v
;               -->-------->H---------
;               |  ^        |        |
;               |  |        |        |
;               |  |        |        |
;               |  |        |        |
;               |  |        v        V
;               |  R<-------T-->U--->P
;               |          Div       |
;               |                    |
;               ----------- Q <-------
;                           |
;                           v
;                          Exit
;
; This is a reducible cycle with a divergent branch at T. Disjoint
; paths eventually join at the header H, which is assigned label H.
; Node P is assigned label U. If the header label were propagated to
; P, it will be incorrectly recgonized as a join. P is always executed
; convergently -- either by threads that reconverged at header H, or
; by threads that diverged at T (and eventually reconverged at H).
; Thus, any PHI at P should not be marked divergent.

define amdgpu_kernel void @header_label_1(i32 %a, i32 %b, i32 %c) {
; CHECK=LABEL: UniformityInfo for function 'header_label_1':
; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK:  depth=1: entries(H) Q P U T R
entry:
  %cond.uni = icmp slt i32 %a, 0
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond.div = icmp slt i32 %tid, 0
  br label %H

H:
  br i1 %cond.uni, label %T, label %P

P:
; CHECK-LABEL: BLOCK P
; CHECK-NOT:   DIVERGENT:   %pp.phi =
; CHECK-NOT: DIVERGENT:      %pp =
  %pp.phi  = phi i32 [ %a, %H], [ %b, %U ]
  %pp = add i32 %b, 1
  br label %Q

Q:
; CHECK-LABEL: BLOCK Q
; CHECK-NOT: DIVERGENT:   %qq =
; CHECK-NOT:   DIVERGENT:   %qq.uni =
  %qq = add i32 %b, 1
  %qq.uni = add i32 %pp.phi, 1
  br i1 %cond.uni, label %exit, label %H

R:
  br label %H

T:
  br i1 %cond.div, label %R, label %U

U:
  br label %P

exit:
; CHECK-LABEL: BLOCK exit
; CHECK: DIVERGENT:     %ee.div =
; CHECK-NOT: DIVERGENT:     %ee =
  %ee.div =  add i32 %qq.uni, 1
  %ee = add i32 %b, 1
  ret void
}

;        entry
;            |
;        --> H1
;        |   | \
;        |   | H2(div)
;        |   \ / \
;        |    B   C
;        ^     \ /
;        \------D
;               |
;               X
;
; This is a reducible cycle with a divergent branch at H2. Disjoint
; paths eventually join at the header D, which is assigned label D.
; Node B is assigned label B. If the header label D were propagated to
; B, it will be incorrectly recgonized as a join. B is always executed
; convergently -- either by threads that reconverged at header H1, or
; by threads that diverge at H2 (and eventually reconverged at H1).
; Thus, any PHI at B should not be marked divergent.

define amdgpu_kernel void @header_label_2(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: UniformityInfo for function 'header_label_2':
; CHECK-NOT: CYCLES ASSUMED DIVERGENT:
; CHECK-NOT: CYCLES WITH DIVERGENT EXIT:
entry:
  %cond.uni = icmp slt i32 %a, 0
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond.div = icmp slt i32 %tid, 0
  br label %H1

H1:
  br i1 %cond.uni, label %B, label %H2

H2:
  br i1 %cond.div, label %B, label %C

B:
; CHECK-LABEL: BLOCK B
; CHECK-NOT: DIVERGENT:     %bb.phi =
  %bb.phi = phi i32 [%a, %H1], [%b, %H2]
  br label %D

C:
  br label %D

D:
; CHECK-LABEL: BLOCK D
; CHECK: DIVERGENT:     %dd.phi =
  %dd.phi = phi i32 [%a, %B], [%b, %C]
  br i1 %cond.uni, label %exit, label %H1

exit:
  %ee.1 = add i32 %dd.phi, 1
  %ee.2 = add i32 %b, 1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
