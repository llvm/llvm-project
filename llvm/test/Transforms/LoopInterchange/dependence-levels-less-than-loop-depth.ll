; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -disable-output -debug 2>&1 | FileCheck %s

;; In the following case, p0 and p1 may alias, so the direction vector must be [* *].
;;
;; void may_alias(float *p0, float *p1) {
;;   for (int i = 0; i < 4; i++)
;;     for (int j = 0; j < 4; j++)
;;       p0[4 * i + j] = p1[4 * j + i];
;; }

; CHECK:      Dependency matrix before interchange:
; CHECK-NEXT: * *
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
define void @may_alias(ptr %p0, ptr %p1) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %idx.0 = getelementptr inbounds [4 x [4 x float]], ptr %p0, i32 0, i32 %j, i32 %i
  %idx.1 = getelementptr inbounds [4 x [4 x float]], ptr %p1, i32 0, i32 %i, i32 %j
  %0 = load float, ptr %idx.0, align 4
  store float %0, ptr %idx.1, align 4
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 4
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 4
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

@A = global [4 x [4 x [4 x float]]] zeroinitializer, align 4
@V = global [4 x float] zeroinitializer, align 4

;; In the following case, there is an all direction dependence for the j-loop
;; and k-loop.
;;
;; for (int i = 0; i < 4; i++)
;;   for (int j = 0; j < 4; j++)
;;     for (int k = 0; k < 4; k++)
;;       V[i] += A[i][j][k];

; CHECK:      Dependency matrix before interchange:
; CHECK-NEXT: = * *
; CHECK-NEXT: Processing InnerLoopId = 2 and OuterLoopId = 1
define void @partial_reduction() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j.header ], [ %k.inc, %for.k ]
  %idx.a = getelementptr inbounds [4 x [4 x [4 x float]]], ptr @A, i32 0, i32 %i, i32 %j, i32 %k
  %idx.v = getelementptr inbounds [4 x float], ptr @V, i32 0, i32 %i
  %0 = load float, ptr %idx.a, align 4
  %1 = load float, ptr %idx.v, align 4
  %add = fadd fast float %0, %1
  store float %add, ptr %idx.v, align 4
  %k.inc = add nuw nsw i32 %k, 1
  %cmp.k = icmp slt i32 %k.inc, 4
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 4
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 4
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}
