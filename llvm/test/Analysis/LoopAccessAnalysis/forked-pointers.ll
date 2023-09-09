; RUN: opt -disable-output -passes='print<access-info>' %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -passes='print<access-info>' -max-forked-scev-depth=2 %s 2>&1 | FileCheck -check-prefix=RECURSE %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: function 'forked_ptrs_simple':
; CHECK-NEXT:  loop:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
; CHECK-NEXT:        %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %select = select i1 %cmp, ptr %gep.1, ptr %gep.2
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
; CHECK-NEXT:        %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %select = select i1 %cmp, ptr %gep.1, ptr %gep.2
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]
; CHECK-NEXT:        (Low: %Dest High: (400 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%loop>
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%loop>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Base1 High: (400 + %Base1))
; CHECK-NEXT:          Member: {%Base1,+,4}<nw><%loop>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %Base2 High: (400 + %Base2))
; CHECK-NEXT:          Member: {%Base2,+,4}<nw><%loop>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define void @forked_ptrs_simple(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr %Dest) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
  %l.Dest = load float, ptr %gep.Dest
  %cmp = fcmp une float %l.Dest, 0.0
  %gep.1 = getelementptr inbounds float, ptr %Base1, i64 %iv
  %gep.2 = getelementptr inbounds float, ptr %Base2, i64 %iv
  %select = select i1 %cmp, ptr %gep.1, ptr %gep.2
  %sink = load float, ptr %select, align 4
  store float %sink, ptr %gep.Dest, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 100
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: function 'forked_ptrs_different_base_same_offset':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (400 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %Base2 High: (400 + %Base2))
; CHECK-NEXT:          Member: {%Base2,+,4}<nw><%for.body>
; CHECK-NEXT:      Group [[G4]]:
; CHECK-NEXT:        (Low: %Base1 High: (400 + %Base1))
; CHECK-NEXT:          Member: {%Base1,+,4}<nw><%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:   Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:   SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:   Expressions re-written:

;; We have a limit on the recursion depth for finding a loop invariant or
;; addrec term; confirm we won't exceed that depth by forcing a lower
;; limit via -max-forked-scev-depth=2
; RECURSE-LABEL: Loop access info in function 'forked_ptrs_same_base_different_offset':
; RECURSE-NEXT:   for.body:
; RECURSE-NEXT:     Report: cannot identify array bounds
; RECURSE-NEXT:     Dependences:
; RECURSE-NEXT:     Run-time memory checks:
; RECURSE-NEXT:     Grouped accesses:
; RECURSE-EMPTY:
; RECURSE-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; RECURSE-NEXT:     SCEV assumptions:
; RECURSE-EMPTY:
; RECURSE-NEXT:     Expressions re-written:

;;;; Derived from the following C code
;; void forked_ptrs_different_base_same_offset(float *A, float *B, float *C, int *D) {
;;   for (int i=0; i<100; i++) {
;;     if (D[i] != 0) {
;;       C[i] = A[i];
;;     } else {
;;       C[i] = B[i];
;;     }
;;   }
;; }

define dso_local void @forked_ptrs_different_base_same_offset(ptr nocapture readonly nonnull %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %indvars.iv
  %.sink = load float, ptr %.sink.in, align 4
  %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %.sink, ptr %1, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_different_base_same_offset_64b':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %1 = getelementptr inbounds double, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds double, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds double, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds double, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds double, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (800 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,8}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %Base2 High: (800 + %Base2))
; CHECK-NEXT:          Member: {%Base2,+,8}<nw><%for.body>
; CHECK-NEXT:      Group [[G4]]:
; CHECK-NEXT:        (Low: %Base1 High: (800 + %Base1))
; CHECK-NEXT:          Member: {%Base1,+,8}<nw><%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_different_base_same_offset_64b(ptr nocapture readonly nonnull %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds double, ptr %spec.select, i64 %indvars.iv
  %.sink = load double, ptr %.sink.in, align 8
  %1 = getelementptr inbounds double, ptr %Dest, i64 %indvars.iv
  store double %.sink, ptr %1, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_different_base_same_offset_23b':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %1 = getelementptr inbounds i23, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds i23, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds i23, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds i23, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds i23, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (399 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %Base2 High: (399 + %Base2))
; CHECK-NEXT:          Member: {%Base2,+,4}<nw><%for.body>
; CHECK-NEXT:      Group [[G4]]:
; CHECK-NEXT:        (Low: %Base1 High: (399 + %Base1))
; CHECK-NEXT:          Member: {%Base1,+,4}<nw><%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_different_base_same_offset_23b(ptr nocapture readonly nonnull %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds i23, ptr %spec.select, i64 %indvars.iv
  %.sink = load i23, ptr %.sink.in
  %1 = getelementptr inbounds i23, ptr %Dest, i64 %indvars.iv
  store i23 %.sink, ptr %1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_different_base_same_offset_6b':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %1 = getelementptr inbounds i6, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds i6, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds i6, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds i6, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds i6, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (100 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,1}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %Base2 High: (100 + %Base2))
; CHECK-NEXT:          Member: {%Base2,+,1}<nw><%for.body>
; CHECK-NEXT:      Group [[G4]]:
; CHECK-NEXT:        (Low: %Base1 High: (100 + %Base1))
; CHECK-NEXT:          Member: {%Base1,+,1}<nw><%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_different_base_same_offset_6b(ptr nocapture readonly nonnull %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds i6, ptr %spec.select, i64 %indvars.iv
  %.sink = load i6, ptr %.sink.in
  %1 = getelementptr inbounds i6, ptr %Dest, i64 %indvars.iv
  store i6 %.sink, ptr %1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_different_base_same_offset_possible_poison':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4:.+]]):
; CHECK-NEXT:        %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %indvars.iv
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (400 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %Base2 High: (400 + %Base2))
; CHECK-NEXT:          Member: {%Base2,+,4}<nw><%for.body>
; CHECK-NEXT:      Group [[G4]]:
; CHECK-NEXT:        (Low: %Base1 High: (400 + %Base1))
; CHECK-NEXT:          Member: {%Base1,+,4}<nw><%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:   Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:   SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:   Expressions re-written:

define dso_local void @forked_ptrs_different_base_same_offset_possible_poison(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds, i1 %c) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %latch ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %indvars.iv
  %.sink = load float, ptr %.sink.in, align 4
  %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  br i1 %c, label %then, label %latch

then:
  store float %.sink, ptr %1, align 4
  br label %latch

latch:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_same_base_different_offset':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: cannot identify array bounds
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:     SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:     Expressions re-written:

;;;; Derived from the following C code
;; void forked_ptrs_same_base_different_offset(float *A, float *B, int *C) {
;;   int offset;
;;   for (int i = 0; i < 100; i++) {
;;     if (C[i] != 0)
;;       offset = i;
;;     else
;;       offset = i+1;
;;     B[i] = A[offset];
;;   }
;; }

define dso_local void @forked_ptrs_same_base_different_offset(ptr nocapture readonly %Base, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %i.014 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %add = add nuw nsw i32 %i.014, 1
  %1 = trunc i64 %indvars.iv to i32
  %offset.0 = select i1 %cmp1.not, i32 %add, i32 %1
  %idxprom213 = zext i32 %offset.0 to i64
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %idxprom213
  %2 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %2, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_add_to_offset'
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
; CHECK-NEXT:        %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (400 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: ((4 * %extra_offset) + %Base) High: (404 + (4 * %extra_offset) + %Base))
; CHECK-NEXT:          Member: {(4 + (4 * %extra_offset) + %Base),+,4}<%for.body>
; CHECK-NEXT:          Member: {((4 * %extra_offset) + %Base),+,4}<%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_add_to_offset(ptr nocapture readonly %Base, ptr nocapture %Dest, ptr nocapture readonly %Preds, i64 %extra_offset) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %sel = select i1 %cmp.not, i64 %indvars.iv.next, i64 %indvars.iv
  %offset = add nuw nsw i64 %sel, %extra_offset
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_sub_from_offset'
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
; CHECK-NEXT:        %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (400 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: ((-4 * %extra_offset) + %Base) High: (404 + (-4 * %extra_offset) + %Base))
; CHECK-NEXT:          Member: {(4 + (-4 * %extra_offset) + %Base),+,4}<%for.body>
; CHECK-NEXT:          Member: {((-4 * %extra_offset) + %Base),+,4}<%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_sub_from_offset(ptr nocapture readonly %Base, ptr nocapture %Dest, ptr nocapture readonly %Preds, i64 %extra_offset) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %sel = select i1 %cmp.not, i64 %indvars.iv.next, i64 %indvars.iv
  %offset = sub nuw nsw i64 %sel, %extra_offset
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_add_sub_offset'
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G3:.+]]):
; CHECK-NEXT:        %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
; CHECK-NEXT:        %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %Dest High: (400 + %Dest))
; CHECK-NEXT:          Member: {%Dest,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (400 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: ((4 * %to_add) + (-4 * %to_sub) + %Base) High: (404 + (4 * %to_add) + (-4 * %to_sub) + %Base))
; CHECK-NEXT:          Member: {(4 + (4 * %to_add) + (-4 * %to_sub) + %Base),+,4}<%for.body>
; CHECK-NEXT:          Member: {((4 * %to_add) + (-4 * %to_sub) + %Base),+,4}<%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_add_sub_offset(ptr nocapture readonly %Base, ptr nocapture %Dest, ptr nocapture readonly %Preds, i64 %to_add, i64 %to_sub) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %sel = select i1 %cmp.not, i64 %indvars.iv.next, i64 %indvars.iv
  %add = add nuw nsw i64 %sel, %to_add
  %offset = sub nuw nsw i64 %add, %to_sub
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

;;;; Cases that can be handled by a forked pointer but are not currently allowed.

; CHECK-LABEL: function 'forked_ptrs_mul_by_offset'
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Report: cannot identify array bounds
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define dso_local void @forked_ptrs_mul_by_offset(ptr nocapture readonly %Base, ptr nocapture %Dest, ptr nocapture readonly %Preds, i64 %extra_offset) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %sel = select i1 %cmp.not, i64 %indvars.iv.next, i64 %indvars.iv
  %offset = mul nuw nsw i64 %sel, %extra_offset
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: function 'forked_ptrs_uniform_and_strided_forks':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Report: cannot identify array bounds
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

;;;; Derived from forked_ptrs_same_base_different_offset with a manually
;;;; added uniform offset and a mul to provide a stride

define dso_local void @forked_ptrs_uniform_and_strided_forks(float* nocapture readonly %Base, float* nocapture %Dest, i32* nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %i.014 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %add = add nuw nsw i32 %i.014, 1
  %1 = trunc i64 %indvars.iv to i32
  %mul = mul i32 %1, 3
  %offset.0 = select i1 %cmp1.not, i32 4, i32 %mul
  %idxprom213 = sext i32 %offset.0 to i64
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %idxprom213
  %2 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %2, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL:  function 'forked_ptrs_gather_and_contiguous_forks':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: cannot identify array bounds
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:     SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:     Expressions re-written:

;;;; Derived from forked_ptrs_same_base_different_offset with a gather
;;;; added using Preds as an index array in addition to the per-iteration
;;;; condition.

define dso_local void @forked_ptrs_gather_and_contiguous_forks(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %arrayidx9 = getelementptr inbounds float, ptr %Base2, i64 %indvars.iv
  %idxprom4 = sext i32 %0 to i64
  %arrayidx5 = getelementptr inbounds float, ptr %Base1, i64 %idxprom4
  %.sink.in = select i1 %cmp1.not, ptr %arrayidx9, ptr %arrayidx5
  %.sink = load float, ptr %.sink.in, align 4
  %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %.sink, ptr %1, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

;; We don't currently handle a fork in both the base and the offset of a
;; GEP instruction.

; CHECK-LABEL: Loop access info in function 'forked_ptrs_two_forks_gep':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: cannot identify array bounds
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:     SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:     Expressions re-written:

define dso_local void @forked_ptrs_two_forks_gep(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %offset = select i1 %cmp1.not, i64 %indvars.iv.next, i64 %indvars.iv
  %.sink.in = getelementptr inbounds float, ptr %spec.select, i64 %offset
  %.sink = load float, ptr %.sink.in, align 4
  %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %.sink, ptr %1, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

;; We don't handle forks as children of a select

; CHECK-LABEL: Loop access info in function 'forked_ptrs_two_select':
; CHECK-NEXT:  loop:
; CHECK-NEXT:    Report: cannot identify array bounds
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:

define void @forked_ptrs_two_select(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr nocapture readonly %Base3, ptr %Dest) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
  %l.Dest = load float, ptr %gep.Dest
  %cmp = fcmp une float %l.Dest, 0.0
  %cmp1 = fcmp une float %l.Dest, 1.0
  %gep.1 = getelementptr inbounds float, ptr %Base1, i64 %iv
  %gep.2 = getelementptr inbounds float, ptr %Base2, i64 %iv
  %gep.3 = getelementptr inbounds float, ptr %Base3, i64 %iv
  %select = select i1 %cmp, ptr %gep.1, ptr %gep.2
  %select1 = select i1 %cmp1, ptr %select, ptr %gep.3
  %sink = load float, ptr %select1, align 4
  store float %sink, ptr %gep.Dest, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 100
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

;; We don't yet handle geps with more than 2 operands
; CHECK-LABEL: Loop access info in function 'forked_ptrs_too_many_gep_ops':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: cannot identify array bounds
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:     SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:     Expressions re-written:

define void @forked_ptrs_too_many_gep_ops(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds [1000 x float], ptr %spec.select, i64 0, i64 %indvars.iv
  %.sink = load float, ptr %.sink.in, align 4
  %1 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %.sink, ptr %1, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

;; We don't currently handle vector GEPs
; CHECK-LABEL: Loop access info in function 'forked_ptrs_vector_gep':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: cannot identify array bounds
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:     SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:     Expressions re-written:

define void @forked_ptrs_vector_gep(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds <4 x float>, ptr %spec.select, i64 %indvars.iv
  %.sink = load <4 x float>, ptr %.sink.in, align 4
  %1 = getelementptr inbounds <4 x float>, ptr %Dest, i64 %indvars.iv
  store <4 x float> %.sink, ptr %1, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-LABEL: Loop access info in function 'sc_add_expr_ice':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Memory dependences are safe with run-time checks
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Check 0:
; CHECK-NEXT:       Comparing group ([[G1:.+]]):
; CHECK-NEXT:       ptr %Base1
; CHECK-NEXT:       Against group ([[G2:.+]]):
; CHECK-NEXT:         %fptr = getelementptr inbounds double, ptr %Base2, i64 %sel
; CHECK-NEXT:     Grouped accesses:
; CHECK-NEXT:       Group [[G1]]:
; CHECK-NEXT:         (Low: %Base1 High: (8 + %Base1))
; CHECK-NEXT:           Member: %Base1
; CHECK-NEXT:       Group [[G2]]:
; CHECK-NEXT:         (Low: %Base2 High: ((8 * %N) + %Base2))
; CHECK-NEXT:           Member: {%Base2,+,8}<%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:     Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:     SCEV assumptions:
; CHECK-NEXT:     {0,+,1}<%for.body> Added Flags: <nusw>
; CHECK-EMPTY:
; CHECK-NEXT:     Expressions re-written:
; CHECK-NEXT:     [PSE]  %fptr = getelementptr inbounds double, ptr %Base2, i64 %sel:
; CHECK-NEXT:       ((8 * (zext i32 {0,+,1}<%for.body> to i64))<nuw><nsw> + %Base2)<nuw>
; CHECK-NEXT:       --> {%Base2,+,8}<%for.body>

;;; The following test caused an ICE with the initial forked pointers work.
;;; One fork is loop invariant (%Base2 + 0), the other is an scAddExpr that
;;; contains an scAddRecExpr inside it:
;;;   ((8 * (zext i32 {0,+,1}<%for.body> to i64))<nuw><nsw> + %Base2)<nuw>
;;;
;;; RtCheck::insert was expecting either loop invariant or SAR, so asserted
;;; on a plain scAddExpr. For now we restrict to loop invariant or SAR
;;; forks only, but we should be able to do better.

define void @sc_add_expr_ice(ptr %Base1, ptr %Base2, i64 %N) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.trunc = trunc i64 %iv to i32
  store double 0.000000e+00, ptr %Base1, align 8
  %iv.zext = zext i32 %iv.trunc to i64
  %sel = select i1 true, i64 %iv.zext, i64 0
  %fptr = getelementptr inbounds double, ptr %Base2, i64 %sel
  %dummy.load = load double, ptr %fptr, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

define void @forked_ptrs_with_different_base(ptr nocapture readonly %Preds, ptr nocapture %a, ptr nocapture %b, ptr nocapture readonly %c)  {
; CHECK:       for.body:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[G1:.+]]):
; CHECK-NEXT:        %arrayidx7 = getelementptr inbounds double, ptr %.sink, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[G1]]):
; CHECK-NEXT:        %arrayidx7 = getelementptr inbounds double, ptr %.sink, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4:.+]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds double, ptr %0, i64 %indvars.iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[G3:.+]]):
; CHECK-NEXT:        %arrayidx7 = getelementptr inbounds double, ptr %.sink, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G2]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
; CHECK-NEXT:    Check 3:
; CHECK-NEXT:      Comparing group ([[G3]]):
; CHECK-NEXT:        %arrayidx7 = getelementptr inbounds double, ptr %.sink, i64 %indvars.iv
; CHECK-NEXT:      Against group ([[G4]]):
; CHECK-NEXT:        %arrayidx5 = getelementptr inbounds double, ptr %0, i64 %indvars.iv
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[G1]]:
; CHECK-NEXT:        (Low: %1 High: (63992 + %1))
; CHECK-NEXT:          Member: {%1,+,8}<nw><%for.body>
; CHECK-NEXT:      Group [[G3]]:
; CHECK-NEXT:        (Low: %2 High: (63992 + %2))
; CHECK-NEXT:          Member: {%2,+,8}<nw><%for.body>
; CHECK-NEXT:      Group [[G2]]:
; CHECK-NEXT:        (Low: %Preds High: (31996 + %Preds))
; CHECK-NEXT:          Member: {%Preds,+,4}<nuw><%for.body>
; CHECK-NEXT:      Group [[G4]]:
; CHECK-NEXT:        (Low: %0 High: (63992 + %0))
; CHECK-NEXT:          Member: {%0,+,8}<nw><%for.body>
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
entry:
  %0 = load ptr, ptr %c, align 64
  %1 = load ptr, ptr %a, align 64
  %2 = load ptr, ptr %b, align 64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %3 = load i32, ptr %arrayidx, align 4
  %cmp2.not = icmp eq i32 %3, 0
  br i1 %cmp2.not, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds double, ptr %0, i64 %indvars.iv
  %4 = load double, ptr %arrayidx5, align 8
  %add = fadd fast double %4, 1.000000e+00
  br label %for.inc

if.else:                                          ; preds = %for.body
  %5 = mul nuw nsw i64 %indvars.iv, %indvars.iv
  %6 = trunc i64 %5 to i32
  %conv8 = sitofp i32 %6 to double
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %.sink = phi ptr [ %1, %if.then ], [ %2, %if.else ]
  %add.sink = phi double [ %add, %if.then ], [ %conv8, %if.else ]
  %arrayidx7 = getelementptr inbounds double, ptr %.sink, i64 %indvars.iv
  store double %add.sink, ptr %arrayidx7, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 7999
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Negative test: the operator number of PhiNode is not 2.
define void @forked_ptrs_with_different_base3(ptr nocapture readonly %Preds, ptr nocapture %a, ptr nocapture %b, ptr nocapture readonly %c)  {
; CHECK:       for.body:
; CHECK-NEXT:    Report: cannot identify array bounds
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
entry:
  %ld.c = load ptr, ptr %c, align 64
  %ld.a = load ptr, ptr %a, align 64
  %ld.b = load ptr, ptr %b, align 64
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %ld.preds = load i32, ptr %arrayidx, align 4
  switch i32 %ld.preds, label %if.else [
    i32 0, label %if.br0
	i32 1, label %if.br1
  ]

if.br0:                                          ; preds = %for.body
  br label %for.inc

if.br1:                                          ; preds = %for.body
  br label %for.inc

if.else:                                         ; preds = %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.br1, %if.br0
  %.sink = phi ptr [ %ld.a, %if.br0 ], [ %ld.b, %if.br1 ], [ %ld.c, %if.else ]
  %arrayidx7 = getelementptr inbounds double, ptr %.sink, i64 %indvars.iv
  store double 1.000000e+00, ptr %arrayidx7, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 7999
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void
}