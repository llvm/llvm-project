; RUN: opt < %s -passes=loop-interchange -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

; This is a collection of cases where the loops were incorrectly interchanged
; due to mishandling of scalar dependencies.

;; The original code is as follows, with some simplifications from the one in
;; the issue #46867. The interchange is illegal due to the dependency of `s`.
;;
;; void issue46867(int *s, int c, int ff[4][9]) {
;;   for (int d = 0; d <= 2; d++) {
;;     for (int e = 0; e <= 2; e++) {
;;       if ((long)(ff[e][d] && (*s = 3), c) % 4073709551606)
;;         ++*s;
;;     }
;;   }
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        issue46867
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        issue46867
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
define void @issue46867(ptr noundef captures(none) %s, i32 noundef %c, ptr noundef readonly captures(none) %ff) {
entry:
  %tobool7.not = icmp eq i32 %c, 0
  br i1 %tobool7.not, label %for.cond1.preheader.us.preheader, label %entry.split

for.cond1.preheader.us.preheader:
  br label %for.cond1.preheader.us

for.cond1.preheader.us:
  %indvars.iv31 = phi i64 [ 0, %for.cond1.preheader.us.preheader ], [ %indvars.iv.next32, %for.cond.cleanup3.split.us.us ]
  br label %for.body4.us.us

for.body4.us.us:
  %indvars.iv27 = phi i64 [ %indvars.iv.next28, %land.end.us.us ], [ 0, %for.cond1.preheader.us ]
  %arrayidx6.us.us = getelementptr inbounds nuw [9 x i32], ptr %ff, i64 %indvars.iv27, i64 %indvars.iv31
  %0 = load i32, ptr %arrayidx6.us.us, align 4
  %tobool.not.us.us = icmp eq i32 %0, 0
  br i1 %tobool.not.us.us, label %land.end.us.us, label %land.rhs.us.us

land.rhs.us.us:
  store i32 3, ptr %s, align 4
  br label %land.end.us.us

land.end.us.us:
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1
  %exitcond30 = icmp ne i64 %indvars.iv.next28, 3
  br i1 %exitcond30, label %for.body4.us.us, label %for.cond.cleanup3.split.us.us

for.cond.cleanup3.split.us.us:
  %indvars.iv.next32 = add nuw nsw i64 %indvars.iv31, 1
  %exitcond34 = icmp ne i64 %indvars.iv.next32, 3
  br i1 %exitcond34, label %for.cond1.preheader.us, label %for.cond.cleanup.loopexit

entry.split:
  %s.promoted19 = load i32, ptr %s, align 4
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv23 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next24, %for.cond.cleanup3.split ]
  %s.promoted20 = phi i32 [ %s.promoted19, %entry.split ], [ %inc.lcssa, %for.cond.cleanup3.split ]
  br label %for.body4

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup.loopexit21:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.cond.cleanup3.split:
  %inc.lcssa = phi i32 [ %inc, %land.end ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond26 = icmp ne i64 %indvars.iv.next24, 3
  br i1 %exitcond26, label %for.cond1.preheader, label %for.cond.cleanup.loopexit21

for.body4:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %land.end ]
  %1 = phi i32 [ %s.promoted20, %for.cond1.preheader ], [ %inc, %land.end ]
  %arrayidx6 = getelementptr inbounds nuw [9 x i32], ptr %ff, i64 %indvars.iv, i64 %indvars.iv23
  %2 = load i32, ptr %arrayidx6, align 4
  %tobool.not = icmp eq i32 %2, 0
  br i1 %tobool.not, label %land.end, label %land.rhs

land.rhs:
  store i32 3, ptr %s, align 4
  br label %land.end

land.end:
  %3 = phi i32 [ 3, %land.rhs ], [ %1, %for.body4 ]
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %s, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 3
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3.split
}


;; The original code is as follows, with some simplifications from the one in
;; the issue #47401. The interchange is illegal due to the dependency of `e`.
;;
;; void issue47401(int *e, int bb[][8]) {
;;   for (int c = 0; c <= 7; c++)
;;     for (int d = 4; d; d--)
;;       bb[d + 2][c] && (*e = bb[d][0]);
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        issue47401
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
define void @issue47401(ptr noundef writeonly captures(none) %e, ptr noundef readonly captures(none) %bb) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv22 = phi i64 [ 0, %entry ], [ %indvars.iv.next23, %for.cond.cleanup2 ]
  br label %for.body3

for.cond.cleanup:
  ret void

for.cond.cleanup2:
  %indvars.iv.next23 = add nuw nsw i64 %indvars.iv22, 1
  %exitcond = icmp ne i64 %indvars.iv.next23, 8
  br i1 %exitcond, label %for.cond1.preheader, label %for.cond.cleanup

for.body3:
  %indvars.iv = phi i64 [ 4, %for.cond1.preheader ], [ %indvars.iv.next, %land.end ]
  %0 = getelementptr [8 x i32], ptr %bb, i64 %indvars.iv
  %arrayidx = getelementptr i8, ptr %0, i64 64
  %arrayidx5 = getelementptr inbounds nuw [8 x i32], ptr %arrayidx, i64 0, i64 %indvars.iv22
  %1 = load i32, ptr %arrayidx5, align 4
  %tobool6.not = icmp eq i32 %1, 0
  br i1 %tobool6.not, label %land.end, label %land.rhs

land.rhs:
  %2 = load i32, ptr %0, align 4
  store i32 %2, ptr %e, align 4
  br label %land.end

land.end:
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %tobool.not = icmp eq i64 %indvars.iv.next, 0
  br i1 %tobool.not, label %for.cond.cleanup2, label %for.body3
}

;; The original code is as follows, with some simplifications from the one in
;; the issue #47295. The interchange is illegal due to the dependency of `f`.
;;
;; void issue47295(int *f, int cc[4][4]) {
;;   for (int i = 0; i <= 3; i++) {
;;     for (int j = 0; j <= 3; j++) {
;;       *f ^= 0x1000;
;;       cc[j][i] = *f;
;;     }
;;   }
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        issue47295
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
define void @issue47295(ptr noundef captures(none) %f, ptr noundef writeonly captures(none) %cc) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv18 = phi i64 [ 0, %entry ], [ %indvars.iv.next19, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %indvars.iv.next19 = add nuw nsw i64 %indvars.iv18, 1
  %exitcond21 = icmp ne i64 %indvars.iv.next19, 4
  br i1 %exitcond21, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %0 = load i32, ptr %f, align 4
  %xor = xor i32 %0, 4096
  store i32 %xor, ptr %f, align 4
  %arrayidx6 = getelementptr inbounds nuw [4 x i32], ptr %cc, i64 %indvars.iv, i64 %indvars.iv18
  store i32 %xor, ptr %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3
}

;; The original code is as follows, with some simplifications from the one in
;; the issue #54176. The interchange is illegal due to the dependency of `aa`.
;;
;; void issue54176(int n, int m, float aa[1024][128], float bb[1024][128], float cc[1024][128]) {
;;   for (int j = 1; j < 128; j++) {
;;     for (int i = 1; i < 1024; i++) {
;;       cc[i][j] = aa[1][j];
;;       aa[1][j-1] += bb[i][j];
;;     }
;;   }
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        issue54176
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
define void @issue54176(i32 noundef %n, i32 noundef %m, ptr noundef captures(none) %aa, ptr noundef readonly captures(none) %bb, ptr noundef writeonly captures(none) %cc) {

entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %aa, i64 512
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv32 = phi i64 [ 1, %entry ], [ %indvars.iv.next33, %for.cond.cleanup3 ]
  %arrayidx5 = getelementptr inbounds nuw [128 x float], ptr %arrayidx, i64 0, i64 %indvars.iv32
  %0 = add nsw i64 %indvars.iv32, -1
  %arrayidx16 = getelementptr inbounds [128 x float], ptr %arrayidx, i64 0, i64 %0
  br label %for.body4

for.cond.cleanup3:
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %exitcond36 = icmp ne i64 %indvars.iv.next33, 128
  br i1 %exitcond36, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %1 = load float, ptr %arrayidx5, align 4
  %arrayidx9 = getelementptr inbounds nuw [128 x float], ptr %cc, i64 %indvars.iv, i64 %indvars.iv32
  store float %1, ptr %arrayidx9, align 4
  %arrayidx13 = getelementptr inbounds nuw [128 x float], ptr %bb, i64 %indvars.iv, i64 %indvars.iv32
  %2 = load float, ptr %arrayidx13, align 4
  %3 = load float, ptr %arrayidx16, align 4
  %add = fadd float %2, %3
  store float %add, ptr %arrayidx16, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup:
  ret void
}

;; The original code is as follows, with some simplifications from the one in
;; the issue #116114. The interchange is illegal due to the dependency of `A`.
;;
;; void issue116114(int *A, int x, unsigned N, unsigned M) {
;;   for (unsigned m = 0; m < M; ++m)
;;     for (unsigned i = 0U; i < N - 1; ++i) {
;;       A[i] = A[i + 1] + x;
;;     }
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        issue116114
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
define void @issue116114(ptr noundef captures(none) %A, i32 noundef %x, i32 noundef %N, i32 noundef %M) {
entry:
  %cmp18.not = icmp eq i32 %M, 0
  br i1 %cmp18.not, label %for.cond.cleanup, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:
  %sub = add i32 %N, -1
  %cmp216.not = icmp eq i32 %sub, 0
  br i1 %cmp216.not, label %for.cond1.preheader.preheader, label %for.cond1.preheader.us.preheader

for.cond1.preheader.us.preheader:
  br label %for.cond1.preheader.us

for.cond1.preheader.preheader:
  br label %for.cond.cleanup.loopexit

for.cond1.preheader.us:
  %m.019.us = phi i32 [ %inc9.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %wide.trip.count = zext i32 %sub to i64
  br label %for.body4.us

for.body4.us:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader.us ], [ %indvars.iv.next, %for.body4.us ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx.us = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv.next
  %0 = load i32, ptr %arrayidx.us, align 4
  %add5.us = add nsw i32 %0, %x
  %arrayidx7.us = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add5.us, ptr %arrayidx7.us, align 4
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body4.us, label %for.cond1.for.cond.cleanup3_crit_edge.us

for.cond1.for.cond.cleanup3_crit_edge.us:
  %inc9.us = add nuw i32 %m.019.us, 1
  %exitcond22 = icmp ne i32 %inc9.us, %M
  br i1 %exitcond22, label %for.cond1.preheader.us, label %for.cond.cleanup.loopexit20

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup.loopexit20:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
