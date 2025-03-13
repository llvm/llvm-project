; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -disable-output -debug 2>&1 | FileCheck %s

@a = dso_local global [20 x [20 x [20 x i32]]] zeroinitializer, align 4
@aa = dso_local global [256 x [256 x float]] zeroinitializer, align 64
@bb = dso_local global [256 x [256 x float]] zeroinitializer, align 64

;;  for (int nl=0;nl<100;++nl)
;;    for (int i=0;i<256;++i)
;;      for (int j=1;j<256;++j)
;;        aa[j][i] = aa[j-1][i] + bb[j][i];
;;
;; The direction vector of `aa` is [* = >]. We can interchange the innermost
;; two loops, The direction vector after interchanging will be [* > =].

; CHECK: Dependency matrix before interchange:
; CHECK-NEXT: * = >
; CHECK-NEXT: * = =
; CHECK-NEXT: Processing InnerLoopId = 2 and OuterLoopId = 1
; CHECK-NEXT: Checking if loops are tightly nested
; CHECK-NEXT: Checking instructions in Loop header and Loop latch
; CHECK-NEXT: Loops are perfectly nested
; CHECK-NEXT: Loops are legal to interchange
; CHECK: Dependency matrix after interchange:
; CHECK-NEXT: * > =
; CHECK-NEXT: * = =

define void @all_eq_gt() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %nl.036 = phi i32 [ 0, %entry ], [ %inc23, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup3:
  %inc23 = add nuw nsw i32 %nl.036, 1
  %exitcond43 = icmp ne i32 %inc23, 100
  br i1 %exitcond43, label %for.cond1.preheader, label %for.cond.cleanup

for.cond.cleanup7:
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1
  %exitcond42 = icmp ne i64 %indvars.iv.next40, 256
  br i1 %exitcond42, label %for.cond5.preheader, label %for.cond.cleanup3

for.body8:
  %indvars.iv = phi i64 [ 1, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx10 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %0, i64 %indvars.iv39
  %1 = load float, ptr %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [256 x [256 x float]], ptr @bb, i64 0, i64 %indvars.iv, i64 %indvars.iv39
  %2 = load float, ptr %arrayidx14, align 4
  %add = fadd fast float %2, %1
  %arrayidx18 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %indvars.iv, i64 %indvars.iv39
  store float %add, ptr %arrayidx18, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.body8, label %for.cond.cleanup7

for.cond5.preheader:
  %indvars.iv39 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next40, %for.cond.cleanup7 ]
  br label %for.body8

for.cond.cleanup:
  ret void
}

;;  for (int i=0;i<256;++i)
;;    for (int j=1;j<256;++j)
;;      aa[j][i] = aa[j-1][255-i] + bb[j][i];
;;
;; The direction vector of `aa` is [* >]. We cannot interchange the loops
;; because we must handle a `*` dependence conservatively.

; CHECK: Dependency matrix before interchange:
; CHECK-NEXT: * >
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Failed interchange InnerLoopId = 1 and OuterLoopId = 0 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.

define void @all_gt() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv31 = phi i64 [ 0, %entry ], [ %indvars.iv.next32, %for.cond.cleanup3 ]
  %0 = sub nuw nsw i64 255, %indvars.iv31
  br label %for.body4

for.cond.cleanup3:
  %indvars.iv.next32 = add nuw nsw i64 %indvars.iv31, 1
  %exitcond35 = icmp ne i64 %indvars.iv.next32, 256
  br i1 %exitcond35, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %1 = add nsw i64 %indvars.iv, -1
  %arrayidx7 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %1, i64 %0
  %2 = load float, ptr %arrayidx7, align 4
  %arrayidx11 = getelementptr inbounds [256 x [256 x float]], ptr @bb, i64 0, i64 %indvars.iv, i64 %indvars.iv31
  %3 = load float, ptr %arrayidx11, align 4
  %add = fadd fast float %3, %2
  %arrayidx15 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %indvars.iv, i64 %indvars.iv31
  store float %add, ptr %arrayidx15, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup:
  ret void
}

;;  for (int i=0;i<255;++i)
;;    for (int j=1;j<256;++j)
;;      aa[j][i] = aa[j-1][i+1] + bb[j][i];
;;
;; The direciton vector of `aa` is [< >]. We cannot interchange the loops
;; because the read/write order for `aa` cannot be changed.

; CHECK: Dependency matrix before interchange:
; CHECK-NEXT: < >
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Failed interchange InnerLoopId = 1 and OuterLoopId = 0 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.

define void @lt_gt() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv31 = phi i64 [ 0, %entry ], [ %indvars.iv.next32, %for.cond.cleanup3 ]
  %indvars.iv.next32 = add nuw nsw i64 %indvars.iv31, 1
  br label %for.body4

for.cond.cleanup3:
  %exitcond34 = icmp ne i64 %indvars.iv.next32, 255
  br i1 %exitcond34, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx6 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %0, i64 %indvars.iv.next32
  %1 = load float, ptr %arrayidx6, align 4
  %arrayidx10 = getelementptr inbounds [256 x [256 x float]], ptr @bb, i64 0, i64 %indvars.iv, i64 %indvars.iv31
  %2 = load float, ptr %arrayidx10, align 4
  %add11 = fadd fast float %2, %1
  %arrayidx15 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %indvars.iv, i64 %indvars.iv31
  store float %add11, ptr %arrayidx15, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup:
  ret void
}

;;  for (int i=0;i<20;i++)
;;    for (int j=0;j<20;j++)
;;      for (int k=1;k<20;k++)
;;        a[i][j][k] = a[i][5][k-1];
;;
;; The direction vector of `a` is [= * >]. We cannot interchange all the loops.

; CHECK: Dependency matrix before interchange:
; CHECK-NEXT: = * >
; CHECK-NEXT: Processing InnerLoopId = 2 and OuterLoopId = 1
; CHECK-NEXT: Failed interchange InnerLoopId = 2 and OuterLoopId = 1 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.
; CHECK-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Failed interchange InnerLoopId = 1 and OuterLoopId = 0 due to dependence
; CHECK-NEXT: Not interchanging loops. Cannot prove legality.

define void @eq_all_gt() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv44 = phi i64 [ 0, %entry ], [ %indvars.iv.next45, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup3:
  %indvars.iv.next45 = add nuw nsw i64 %indvars.iv44, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next45, 20
  br i1 %exitcond47, label %for.cond1.preheader, label %for.cond.cleanup

for.cond.cleanup7:
  %indvars.iv.next41 = add nuw nsw i64 %indvars.iv40, 1
  %exitcond43 = icmp ne i64 %indvars.iv.next41, 20
  br i1 %exitcond43, label %for.cond5.preheader, label %for.cond.cleanup3

for.body8:
  %indvars.iv = phi i64 [ 1, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx11 = getelementptr inbounds [20 x [20 x [20 x i32]]], ptr @a, i64 0, i64 %indvars.iv44, i64 5, i64 %0
  %1 = load i32, ptr %arrayidx11, align 4
  %arrayidx17 = getelementptr inbounds nuw [20 x [20 x [20 x i32]]], ptr @a, i64 0, i64 %indvars.iv44, i64 %indvars.iv40, i64 %indvars.iv
  store i32 %1, ptr %arrayidx17, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 20
  br i1 %exitcond, label %for.body8, label %for.cond.cleanup7

for.cond5.preheader:
  %indvars.iv40 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next41, %for.cond.cleanup7 ]
  br label %for.body8

for.cond.cleanup:
  ret void
}
