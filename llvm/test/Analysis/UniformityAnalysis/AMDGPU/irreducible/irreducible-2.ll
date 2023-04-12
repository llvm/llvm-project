; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

define amdgpu_kernel void @cycle_diverge_enter(i32 %n, i32 %a, i32 %b) #0 {
;      entry(div)
;      /   \
;     H <-> B
;           |
;           X
; CHECK-LABEL: for function 'cycle_diverge_enter':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %div.cond, label %B, label %H  ; divergent branch

H:
  %div.merge.h = phi i32 [ 0, %entry ], [ %b, %B ]
  br label %B
; CHECK: DIVERGENT: %div.merge.h

B:
  %div.merge.b = phi i32 [ %a, %H ], [1, %entry ]
  %div.cond.b = icmp sgt i32 %div.merge.b, 0
  %div.b.inc = add i32 %b, 1
  br i1 %div.cond, label %X, label %H ; divergent branch
; CHECK: DIVERGENT: %div.merge.b

X:
  %div.use = add i32 %div.merge.b, 1
  ret void
; CHECK: DIVERGENT: %div.use =

}

define amdgpu_kernel void @cycle_diverge_exit(i32 %n, i32 %a, i32 %b) #0 {
;      entry
;      /   \
;     H <-> B(div)
;           |
;           X
;
; CHECK-LABEL: for function 'cycle_diverge_exit':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %B, label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %b, %B ]
  br label %B

B:
  %uni.merge.b = phi i32 [ %a, %H ], [1, %entry ]
  %uni.cond.b = icmp sgt i32 %uni.merge.b, 0
  %uni.b.inc = add i32 %b, 1
  br i1 %div.cond, label %X, label %H ; divergent branch

X:
  %div.use = add i32 %uni.merge.b, 1
  ret void
; CHECK: DIVERGENT: %div.use =
}

define amdgpu_kernel void @cycle_reentrance(i32 %n, i32 %a, i32 %b) #0 {
; For this case, threads enter the cycle from C would take C->D->H,
; at the point of H, diverged threads may continue looping in cycle(H-B-D)
; until all threads exit the cycle(H-B-D) and cause temporal divergence
; exiting at edge H->C. We currently do not analyze such kind of inner
; cycle temporal divergence. Instead, we mark all values in the cycle
; being divergent conservatively.
;      entry--\
;       |     |
;  ---> H(div)|
;  |   / \    /
;  |  B   C<--
;  ^   \ /
;  \----D
;       |
;       X
; CHECK-LABEL: for function 'cycle_reentrance':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %H, label %C

H:
  %div.merge.h = phi i32 [ 0, %entry ], [ %b, %D ]
  br i1 %div.cond, label %B, label %C  ; divergent branch

B:
  %div.inc.b = add i32 %div.merge.h, 1
; CHECK: DIVERGENT: %div.inc.b
  br label %D

C:
  %div.merge.c = phi i32 [0, %entry], [%div.merge.h, %H]
  %div.inc.c = add i32 %div.merge.c, 2
; CHECK: DIVERGENT: %div.inc.c
  br label %D

D:
  %div.merge.d = phi i32 [ %div.inc.b, %B ], [ %div.inc.c, %C ]
; CHECK: DIVERGENT: %div.merge.d
  br i1 %uni.cond, label %X, label %H

X:
  ret void
}

define amdgpu_kernel void @cycle_reentrance2(i32 %n, i32 %a, i32 %b) #0 {
; This is mostly the same as cycle_reentrance, the only difference is
; the successor order, thus different dfs visiting order. This is just
; make sure we are doing uniform analysis correctly under different dfs
; order.
;      entry--\
;       |     |
;  ---> H(div)|
;  |   / \    /
;  |  B   C<--
;  ^   \ /
;  \----D
;       |
;       X
; CHECK-LABEL: for function 'cycle_reentrance2':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %C, label %H

H:
  %div.merge.h = phi i32 [ 0, %entry ], [ %b, %D ]
  br i1 %div.cond, label %B, label %C  ; divergent branch

B:
  %div.inc.b = add i32 %div.merge.h, 1
; CHECK: DIVERGENT: %div.inc.b
  br label %D

C:
  %div.merge.c = phi i32 [0, %entry], [%div.merge.h, %H]
  %div.inc.c = add i32 %div.merge.c, 2
; CHECK: DIVERGENT: %div.inc.c
  br label %D

D:
  %div.merge.d = phi i32 [ %div.inc.b, %B ], [ %div.inc.c, %C ]
; CHECK: DIVERGENT: %div.merge.d
  br i1 %uni.cond, label %X, label %H

X:
  ret void
}

define amdgpu_kernel void @cycle_join_dominated_by_diverge(i32 %n, i32 %a, i32 %b) #0 {
; the join-node D is dominated by diverge point H2
;      entry
;       | |
;  --> H1 |
;  |     \|
;  |      H2(div)
;  |     / \
;  |    B   C
;  ^     \ /
;  \------D
;         |
;         X
; CHECK-LABEL: for function 'cycle_join_dominated_by_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %H1, label %H2

H1:
  %uni.merge.h1 = phi i32 [ 0, %entry ], [ %b, %D ]
  br label %H2

H2:
  %uni.merge.h2 = phi i32 [ 0, %entry ], [ %b, %H1 ]
  br i1 %div.cond, label %B, label %C  ; divergent branch

B:
  %uni.inc.b = add i32 %uni.merge.h2, 1
  br label %D

C:
  %uni.inc.c = add i32 %uni.merge.h2, 2
  br label %D

D:
  %div.merge.d = phi i32 [ %uni.inc.b, %B ], [ %uni.inc.c, %C ]
; CHECK: DIVERGENT: %div.merge.d
  br i1 %uni.cond, label %X, label %H1

X:
  ret void
}

define amdgpu_kernel void @cycle_join_dominated_by_entry(i32 %n, i32 %a, i32 %b) #0 {
; the join-node D is dominated by cycle entry H2
;      entry
;       | |
;  --> H1 |
;  |     \|
;  |      H2 -----
;  |      |      |
;  |      A(div) |
;  |     / \     v
;  |    B   C   /
;  ^     \ /   /
;  \------D <-/
;         |
;         X
; CHECK-LABEL: for function 'cycle_join_dominated_by_entry':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %H1, label %H2

H1:
  %uni.merge.h1 = phi i32 [ 0, %entry ], [ %b, %D ]
  br label %H2

H2:
  %uni.merge.h2 = phi i32 [ 0, %entry ], [ %b, %H1 ]
  br i1 %uni.cond, label %A, label %D

A:
  br i1 %div.cond, label %B, label %C  ; divergent branch

B:
  %uni.inc.b = add i32 %uni.merge.h2, 1
  br label %D

C:
  %uni.inc.c = add i32 %uni.merge.h2, 2
  br label %D

D:
  %div.merge.d = phi i32 [ %uni.inc.b, %B ], [ %uni.inc.c, %C ], [%uni.merge.h2, %H2]
; CHECK: DIVERGENT: %div.merge.d
  br i1 %uni.cond, label %X, label %H1

X:
  ret void
}

define amdgpu_kernel void @cycle_join_not_dominated(i32 %n, i32 %a, i32 %b) #0 {
; if H is the header, the sync label propagation may stop at join node D.
; But join node D is not dominated by divergence starting block C, and also
; not dominated by any entries(H/C). So we conservatively mark all the values
; in the cycle divergent for now.
;      entry
;       |  |
;  ---> H  |
;  |   / \ v
;  |  B<--C(div)
;  ^   \ /
;  \----D
;       |
;       X
; CHECK-LABEL: for function 'cycle_join_not_dominated':
; CHECK-NOT: DIVERGENT: %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %C, label %H

H:
  %div.merge.h = phi i32 [ 0, %entry ], [ %b, %D ]
  br i1 %uni.cond, label %B, label %C

B:
  %div.merge.b = phi i32 [ 0, %H ], [ %b, %C ]
  %div.inc.b = add i32 %div.merge.b, 1
; CHECK: DIVERGENT: %div.inc.b
  br label %D

C:
  %div.merge.c = phi i32 [0, %entry], [%div.merge.h, %H]
  %div.inc.c = add i32 %div.merge.c, 2
; CHECK: DIVERGENT: %div.inc.c
  br i1 %div.cond, label %D, label %B  ; divergent branch

D:
  %div.merge.d = phi i32 [ %div.inc.b, %B ], [ %div.inc.c, %C ]
; CHECK: DIVERGENT: %div.merge.d
  br i1 %uni.cond, label %X, label %H

X:
  ret void
}

define amdgpu_kernel void @cycle_join_not_dominated2(i32 %n, i32 %a, i32 %b) #0 {
; This is mostly the same as cycle_join_not_dominated, the only difference is
; the dfs visiting order, so the cycle analysis result is different.
;      entry
;       |  |
;  ---> H  |
;  |   / \ v
;  |  B<--C(div)
;  ^   \ /
;  \----D
;       |
;       X
; CHECK-LABEL: for function 'cycle_join_not_dominated2':
; CHECK-NOT: DIVERGENT: %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %H, label %C

H:
  %div.merge.h = phi i32 [ 0, %entry ], [ %b, %D ]
  br i1 %uni.cond, label %B, label %C

B:
  %div.merge.b = phi i32 [ 0, %H ], [ %b, %C ]
  %div.inc.b = add i32 %div.merge.b, 1
; CHECK: DIVERGENT: %div.inc.b
  br label %D

C:
  %div.merge.c = phi i32 [0, %entry], [%div.merge.h, %H]
  %div.inc.c = add i32 %div.merge.c, 2
; CHECK: DIVERGENT: %div.inc.c
  br i1 %div.cond, label %D, label %B  ; divergent branch

D:
  %div.merge.d = phi i32 [ %div.inc.b, %B ], [ %div.inc.c, %C ]
; CHECK: DIVERGENT: %div.merge.d
  br i1 %uni.cond, label %X, label %H

X:
  ret void
}

define amdgpu_kernel void @natural_loop_two_backedges(i32 %n, i32 %a, i32 %b) #0 {
; FIXME: the uni.merge.h can be viewed as uniform.
; CHECK-LABEL: for function 'natural_loop_two_backedges':

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %B ], [ %uni.inc, %C]
  %uni.inc = add i32 %uni.merge.h, 1
  br label %B

B:
  br i1 %div.cond, label %C, label %H

C:
  br i1 %uni.cond, label %X, label %H

X:
  ret void
}

define amdgpu_kernel void @natural_loop_two_backedges2(i32 %n, i32 %a, i32 %b) #0 {
; FIXME: the uni.merge.h can be viewed as uniform.
; CHECK-LABEL: for function 'natural_loop_two_backedges2':

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %B ], [ %uni.inc, %C]
  %uni.inc = add i32 %uni.merge.h, 1
  br i1 %uni.cond, label %B, label %D

B:
  br i1 %div.cond, label %C, label %H

C:
  br label %H

D:
  br i1 %uni.cond, label %B, label %X

X:
  ret void
}

define amdgpu_kernel void @cycle_enter_nested(i32 %n, i32 %a, i32 %b) #0 {
;
;   entry(div)
;       |   \
;   --> H1   |
;  /    |    |
;  | -> H2   |
;  | |  |    /
;  | \--B <--
;  ^    |
;  \----C
;       |
;       X
; CHECK-LABEL: for function 'cycle_enter_nested':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %div.cond, label %B, label %H1

H1:
  %div.merge.h1 = phi i32 [ 1, %entry ], [ %b, %C ]
  br label %H2
; CHECK: DIVERGENT: %div.merge.h1

H2:
  %div.merge.h2 = phi i32 [ 2, %B ], [ %a, %H1 ]
; CHECK: DIVERGENT: %div.merge.h2
  br label %B

B:
  %div.merge.b = phi i32 [0, %entry], [%a, %H2]
; CHECK: DIVERGENT: %div.merge.b
  br i1 %uni.cond, label %C, label %H2

C:
  br i1 %uni.cond, label %X, label %H1

X:
  ret void
}

define amdgpu_kernel void @cycle_inner_exit_enter(i32 %n, i32 %a, i32 %b) #0 {
;          entry
;        /      \
;       E1-> A-> E2
;       ^    |    \
;       |    E3-> E4
;       |    ^   /
;       |     \ /
;       C <----B
;       |
;       X
; CHECK-LABEL: for function 'cycle_inner_exit_enter':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %E2, label %E1

E1:
  %div.merge.e1 = phi i32 [ 1, %entry ], [ %b, %C ]
  br label %A
; CHECK: DIVERGENT: %div.merge.e1

A:
  br i1 %uni.cond, label %E2, label %E3

E2:
  %div.merge.e2 = phi i32 [ 2, %entry ], [ %a, %A ]
; CHECK: DIVERGENT: %div.merge.e2
  br label %E4

E3:
  %div.merge.e3 = phi i32 [ 0, %A ], [ %b, %B ]
; CHECK: DIVERGENT: %div.merge.e3
  br label %E4

E4:
  %div.merge.e4 = phi i32 [ 0, %E2 ], [ %a, %E3 ]
; CHECK: DIVERGENT: %div.merge.e4
  br label %B

B:
  br i1 %div.cond, label %C, label %E3

C:
  br i1 %uni.cond, label %X, label %E1

X:
  ret void
}

define amdgpu_kernel void @cycle_inner_exit_enter2(i32 %n, i32 %a, i32 %b) #0 {
; This case is almost the same as cycle_inner_exit_enter, with only different
; dfs visiting order, thus different cycle hierarchy.
;          entry
;        /      \
;       E1-> A-> E2
;       ^    |    \
;       |    E3-> E4
;       |    ^   /
;       |     \ /
;       C <----B
;       |
;       X
; CHECK-LABEL: for function 'cycle_inner_exit_enter2':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %E1, label %E2

E1:
  %div.merge.e1 = phi i32 [ 1, %entry ], [ %b, %C ]
  br label %A
; CHECK: DIVERGENT: %div.merge.e1

A:
  br i1 %uni.cond, label %E2, label %E3

E2:
  %div.merge.e2 = phi i32 [ 2, %entry ], [ %a, %A ]
; CHECK: DIVERGENT: %div.merge.e2
  br label %E4

E3:
  %div.merge.e3 = phi i32 [ 0, %A ], [ %b, %B ]
; CHECK: DIVERGENT: %div.merge.e3
  br label %E4

E4:
  %div.merge.e4 = phi i32 [ 0, %E2 ], [ %a, %E3 ]
; CHECK: DIVERGENT: %div.merge.e4
  br label %B

B:
  br i1 %div.cond, label %C, label %E3

C:
  br i1 %uni.cond, label %X, label %E1

X:
  ret void
}

define amdgpu_kernel void @always_uniform() {
; CHECK-LABEL: UniformityInfo for function 'always_uniform':
; CHECK: CYCLES ASSSUMED DIVERGENT:
; CHECK:   depth=1: entries(bb2 bb3)

bb:
  %inst = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 0, i32 0)
  %inst1 = icmp ugt i32 %inst, 0
  br i1 %inst1, label %bb3, label %bb2
; CHECK:   DIVERGENT:   %inst = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 0, i32 0)
; CHECK:   DIVERGENT:   %inst1 = icmp ugt i32 %inst, 0
; CHECK:   DIVERGENT:   br i1 %inst1, label %bb3, label %bb2

bb2:                                              ; preds = %bb3, %bb
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %inst4 = tail call i64 @llvm.amdgcn.icmp.i64.i16(i16 0, i16 0, i32 0)
  %inst5 = trunc i64 %inst4 to i32
  %inst6 = and i32 0, %inst5
  br label %bb2
; CHECK-LABEL: BLOCK bb3
; CHECK-NOT: DIVERGENT: {{.*}} call i64 @llvm.amdgcn.icmp.i64.i16
; CHECK:   DIVERGENT:   %inst5 = trunc i64 %inst4 to i32
; CHECK:   DIVERGENT:   %inst6 = and i32 0, %inst5
}

declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)

declare i64 @llvm.amdgcn.icmp.i64.i16(i16, i16, i32 immarg)

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
