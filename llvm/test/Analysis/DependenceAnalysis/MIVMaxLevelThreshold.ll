; RUN: opt < %s -aa-pipeline=default -passes='print<da>' -da-miv-max-level-threshold=2

;; Check to make sure when MIV tests reach a maximum depth level
;; threshold, the resulting dependence is conservatively correct.

;; for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;     for (int k = 0; k < o; k++)
;;       A[i+j+k] = A[i-j+k] + B[i+j+k];


; CHECK:       Src:  %2 = load float, float* %arrayidx, align 4 --> Dst:  %5 = load float, float* %arrayidx12, align 4
; CHECK-NEXT:  da analyze - none!
; CHECK:       Src:  %2 = load float, float* %arrayidx, align 4 --> Dst:  store float %add13, float* %arrayidx17, align 4
; CHECK-NEXT:  da analyze - anti [* * *|<]!
; CHECK:       Src:  %5 = load float, float* %arrayidx12, align 4 --> Dst:  store float %add13, float* %arrayidx17, align 4
; CHECK-NEXT:  da analyze - none!

define void @foo(float* noalias %A, float* noalias %B, i32 signext %m, i32 signext %n, i32 signext %o) {
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end23

for.body.preheader:                               ; preds = %entry
  %wide.trip.count21 = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc21
  %indvars.iv19 = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next20, %for.inc21 ]
  %cmp23 = icmp sgt i32 %m, 0
  br i1 %cmp23, label %for.body4.preheader, label %for.inc21

for.body4.preheader:                              ; preds = %for.body
  %wide.trip.count17 = zext i32 %m to i64
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.inc18
  %indvars.iv12 = phi i64 [ 0, %for.body4.preheader ], [ %indvars.iv.next13, %for.inc18 ]
  %cmp61 = icmp sgt i32 %o, 0
  br i1 %cmp61, label %for.body8.preheader, label %for.inc18

for.body8.preheader:                              ; preds = %for.body4
  %wide.trip.count = zext i32 %o to i64
  br label %for.body8

for.body8:                                        ; preds = %for.body8.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.body8.preheader ], [ %indvars.iv.next, %for.body8 ]
  %0 = sub nsw i64 %indvars.iv19, %indvars.iv12
  %1 = add nsw i64 %0, %indvars.iv
  %arrayidx = getelementptr inbounds float, float* %A, i64 %1
  %2 = load float, float* %arrayidx, align 4
  %3 = add nuw nsw i64 %indvars.iv19, %indvars.iv12
  %4 = add nuw nsw i64 %3, %indvars.iv
  %arrayidx12 = getelementptr inbounds float, float* %B, i64 %4
  %5 = load float, float* %arrayidx12, align 4
  %add13 = fadd fast float %2, %5
  %6 = add nuw nsw i64 %indvars.iv19, %indvars.iv12
  %7 = add nuw nsw i64 %6, %indvars.iv
  %arrayidx17 = getelementptr inbounds float, float* %A, i64 %7
  store float %add13, float* %arrayidx17, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body8, label %for.inc18.loopexit

for.inc18.loopexit:                               ; preds = %for.body8
  br label %for.inc18

for.inc18:                                        ; preds = %for.inc18.loopexit, %for.body4
  %indvars.iv.next13 = add nuw nsw i64 %indvars.iv12, 1
  %exitcond18 = icmp ne i64 %indvars.iv.next13, %wide.trip.count17
  br i1 %exitcond18, label %for.body4, label %for.inc21.loopexit

for.inc21.loopexit:                               ; preds = %for.inc18
  br label %for.inc21

for.inc21:                                        ; preds = %for.inc21.loopexit, %for.body
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond22 = icmp ne i64 %indvars.iv.next20, %wide.trip.count21
  br i1 %exitcond22, label %for.body, label %for.end23.loopexit

for.end23.loopexit:                               ; preds = %for.inc21
  br label %for.end23

for.end23:                                        ; preds = %for.end23.loopexit, %entry
  ret void
}
