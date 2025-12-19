; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -disable-output -debug 2>&1 | FileCheck %s

@a = dso_local global [256 x [256 x float]] zeroinitializer, align 4
@b = dso_local global [20 x [20 x [20 x i32]]] zeroinitializer, align 4

;;  for (int n = 0; n < 100; ++n)
;;    for (int i = 0; i < 256; ++i)
;;      for (int j = 1; j < 256; ++j)
;;        a[j - 1][i] += a[j][i];
;;
;; The direction vector of `a` is [* = <]. We can interchange the innermost
;; two loops, The direction vector after interchanging will be [* < =].

; CHECK:      Dependency matrix before interchange:
; CHECK-NEXT: * = <
; CHECK-NEXT: * = =
; CHECK-NEXT: Processing InnerLoopId = 2 and OuterLoopId = 1
; CHECK-NEXT: Checking if loops are tightly nested
; CHECK-NEXT: Checking instructions in Loop header and Loop latch
; CHECK-NEXT: Loops are perfectly nested
; CHECK-NEXT: Loops are legal to interchange

define void @all_eq_lt() {
entry:
  br label %for.n.header

for.n.header:
  %n = phi i32 [ 0, %entry ], [ %n.inc, %for.n.latch ]
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %for.n.header ], [ %i.inc, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 1, %for.i.header ], [ %j.inc, %for.j ]
  %j.dec = sub nsw i32 %j, 1
  %idx.store = getelementptr inbounds [256 x [256 x float]], ptr @a, i32 0, i32 %j.dec, i32 %i
  %idx.load = getelementptr inbounds [256 x [256 x float]], ptr @a, i32 0, i32 %j, i32 %i
  %0 = load float, ptr %idx.load, align 4
  %1 = load float, ptr %idx.store, align 4
  %add = fadd fast float %0, %1
  store float %add, ptr %idx.store, align 4
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 256
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 256
  br i1 %cmp.i, label %for.i.header, label %for.n.latch

for.n.latch:
  %n.inc = add nuw nsw i32 %n, 1
  %cmp.n = icmp slt i32 %n.inc, 100
  br i1 %cmp.n, label %for.n.header, label %exit

exit:
  ret void
}

;;  for (int i = 0; i < 256; ++i)
;;    for (int j = 1; j < 256; ++j)
;;      a[j - 1][i] = a[j][255 - i];
;;
;; The direction vector of `a` is [* <]. We cannot interchange the loops
;; because we must handle a `*` dependence conservatively.

; CHECK:      Dependency matrix before interchange:
; CHECK-NEXT: * <
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Failed interchange InnerLoopId = 1 and OuterLoopId = 0 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.

define void @all_lt() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %i.rev = sub nsw i32 255, %i
  br label %for.j

for.j:
  %j = phi i32 [ 1, %for.i.header ], [ %j.inc, %for.j ]
  %j.dec = sub nsw i32 %j, 1
  %idx.store = getelementptr inbounds [256 x [256 x float]], ptr @a, i32 0, i32 %j.dec, i32 %i
  %idx.load = getelementptr inbounds [256 x [256 x float]], ptr @a, i32 0, i32 %j, i32 %i.rev
  %0 = load float, ptr %idx.load, align 4
  store float %0, ptr %idx.store, align 4
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 256
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 256
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}
 
;;  for (int i = 0; i < 255; ++i)
;;    for (int j = 1; j < 256; ++j)
;;      a[j][i] = a[j - 1][i + 1];
;;
;; The direciton vector of `a` is [< >]. We cannot interchange the loops
;; because the read/write order for `a` cannot be changed.

; CHECK:      Dependency matrix before interchange:
; CHECK-NEXT: < >
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Failed interchange InnerLoopId = 1 and OuterLoopId = 0 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.

define void @lt_gt() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %i.inc = add nuw nsw i32 %i, 1
  br label %for.j

for.j:
  %j = phi i32 [ 1, %for.i.header ], [ %j.inc, %for.j ]
  %j.dec = sub nsw i32 %j, 1
  %idx.store = getelementptr inbounds [256 x [256 x float]], ptr @a, i32 0, i32 %j, i32 %i
  %idx.load = getelementptr inbounds [256 x [256 x float]], ptr @a, i32 0, i32 %j.dec, i32 %i.inc
  %0 = load float, ptr %idx.load, align 4
  store float %0, ptr %idx.store, align 4
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 256
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %cmp.i = icmp slt i32 %i.inc, 255
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

;;  for (int i = 0; i < 20; i++)
;;    for (int j = 0; j < 20; j++)
;;      for (int k = 0; k < 19; k++)
;;        b[i][j][k] = b[i][5][k + 1];
;;
;; The direction vector of `b` is [= * *]. We cannot interchange all the loops.

; CHECK:      Dependency matrix before interchange:
; CHECK-NEXT: = * *
; CHECK-NEXT: Processing InnerLoopId = 2 and OuterLoopId = 1
; CHECK-NEXT: Failed interchange InnerLoopId = 2 and OuterLoopId = 1 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Failed interchange InnerLoopId = 1 and OuterLoopId = 0 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.

define void @eq_all_lt() {
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
  %k.inc = add nuw nsw i32 %k, 1
  %idx.store = getelementptr inbounds [20 x [20 x [20 x i32]]], ptr @b, i32 0, i32 %i, i32 %j, i32 %k
  %idx.load = getelementptr inbounds [20 x [20 x [20 x i32]]], ptr @b, i32 0, i32 %i, i32 5, i32 %k.inc
  %0 = load i32, ptr %idx.load, align 4
  store i32 %0, ptr %idx.store, align 4
  %cmp.k = icmp slt i32 %k.inc, 19
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 20
  br i1 %cmp.j, label %for.j.header, label %for.i.latch

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 20
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}
