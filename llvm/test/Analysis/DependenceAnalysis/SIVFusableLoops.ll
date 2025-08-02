; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN:   -da-disable-delinearization-checks | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"

;;  for (long int i = 0; i < n; i++) {
;;    for (long int j = 0; j < n; j++) {
;;      for (long int k = 0; k < n; k++) {
;;        for (long int l = 0; l < n; l++)
;;          A[i][j][k][l] = i;
;;      }
;;      for (long int k = 1; k < n+1; k++) {
;;        for (long int l = 0; l < n; l++)
;;          *B++ = A[i + 4][j + 3][k + 2][l + 1];

define void @SIVFusable(i64 %n, ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  %cmp10 = icmp sgt i64 %n, 0
  br i1 %cmp10, label %for.cond1.preheader.preheader, label %for.end35
  
; CHECK-LABEL: SIVFusable
; CHECK: da analyze - none!
; CHECK: da analyze - flow [-4 -3]! / assuming 2 fused loop(s): [-4 -3 -3 -1]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - output [* * * *]!
  
for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc33
  %B.addr.014 = phi ptr [ %B.addr.1.lcssa, %for.inc33 ], [ %B, %for.cond1.preheader.preheader ]
  %i.013 = phi i64 [ %inc34, %for.inc33 ], [ 0, %for.cond1.preheader.preheader ]
  %cmp28 = icmp sgt i64 %n, 0
  br i1 %cmp28, label %for.cond4.preheader.preheader, label %for.inc33

for.cond4.preheader.preheader:                    ; preds = %for.cond1.preheader
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond4.preheader.preheader, %for.inc30
  %B.addr.110 = phi ptr [ %B.addr.2.lcssa, %for.inc30 ], [ %B.addr.014, %for.cond4.preheader.preheader ]
  %j.09 = phi i64 [ %inc31, %for.inc30 ], [ 0, %for.cond4.preheader.preheader ]
  %cmp53 = icmp sgt i64 %n, 0
  br i1 %cmp53, label %for.cond7.preheader.preheader, label %for.cond15.loopexit

for.cond7.preheader.preheader:                    ; preds = %for.cond4.preheader
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond7.preheader.preheader, %for.inc12
  %k.07 = phi i64 [ %inc13, %for.inc12 ], [ 0, %for.cond7.preheader.preheader ]
  %cmp81 = icmp sgt i64 %n, 0
  br i1 %cmp81, label %for.body9.preheader, label %for.inc12

for.body9.preheader:                              ; preds = %for.cond7.preheader
  br label %for.body9

for.body9:                                        ; preds = %for.body9.preheader, %for.body9
  %l.02 = phi i64 [ %inc11, %for.body9 ], [ 0, %for.body9.preheader ]
  %arrayidx12 = getelementptr inbounds [100 x [100 x [100 x i64]]], ptr %A, i64 %i.013, i64 %j.09, i64 %k.07, i64 %l.02
  store i64 %i.013, ptr %arrayidx12, align 8
  %inc11 = add nsw i64 %l.02, 1
  %exitcond15 = icmp ne i64 %inc11, %n
  br i1 %exitcond15, label %for.body9, label %for.inc12.loopexit

for.inc12.loopexit:                               ; preds = %for.body9
  br label %for.inc12

for.inc12:                                        ; preds = %for.inc12.loopexit, %for.cond7.preheader
  %inc13 = add nsw i64 %k.07, 1
  %exitcond16 = icmp ne i64 %inc13, %n
  br i1 %exitcond16, label %for.cond7.preheader, label %for.cond15.loopexit.loopexit

for.cond15.loopexit.loopexit:                     ; preds = %for.inc12
  br label %for.cond15.loopexit

for.cond15.loopexit:                              ; preds = %for.cond15.loopexit.loopexit, %for.cond4.preheader
  %cmp163 = icmp sgt i64 %n, 0
  br i1 %cmp163, label %for.cond18.preheader.preheader, label %for.inc30

for.cond18.preheader.preheader:                   ; preds = %for.cond15.loopexit
  br label %for.cond18.preheader

for.cond18.preheader:                             ; preds = %for.cond18.preheader.preheader, %for.inc27
  %k14.06 = phi i64 [ %inc28, %for.inc27 ], [ 1, %for.cond18.preheader.preheader ]
  %B.addr.25 = phi ptr [ %B.addr.3.lcssa, %for.inc27 ], [ %B.addr.110, %for.cond18.preheader.preheader ]
  %cmp191 = icmp sgt i64 %n, 0
  br i1 %cmp191, label %for.body20.preheader, label %for.inc27

for.body20.preheader:                             ; preds = %for.cond18.preheader
  br label %for.body20

for.body20:                                       ; preds = %for.body20.preheader, %for.body20
  %l17.04 = phi i64 [ %inc25, %for.body20 ], [ 0, %for.body20.preheader ]
  %B.addr.34 = phi ptr [ %incdec.ptr, %for.body20 ], [ %B.addr.25, %for.body20.preheader ]
  %add = add nsw i64 %l17.04, 1
  %add21 = add nsw i64 %k14.06, 2
  %add22 = add nsw i64 %j.09, 3
  %add23 = add nsw i64 %i.013, 4
  %arrayidx24 = getelementptr inbounds [100 x [100 x [100 x i64]]], ptr %A, i64 %add23, i64 %add22, i64 %add21, i64 %add
  %0 = load i64, ptr %arrayidx24, align 8
  %incdec.ptr = getelementptr inbounds i64, ptr %B.addr.34, i64 1
  store i64 %0, ptr %B.addr.34, align 8
  %inc25 = add nsw i64 %l17.04, 1
  %exitcond = icmp ne i64 %inc25, %n
  br i1 %exitcond, label %for.body20, label %for.inc27.loopexit

for.inc27.loopexit:                               ; preds = %for.body20
  %scevgep = getelementptr i64, ptr %B.addr.25, i64 %n
  br label %for.inc27

for.inc27:                                        ; preds = %for.inc27.loopexit, %for.cond18.preheader
  %B.addr.3.lcssa = phi ptr [ %B.addr.25, %for.cond18.preheader ], [ %scevgep, %for.inc27.loopexit ]
  %inc28 = add nsw i64 %k14.06, 1
  %inc29 = add nsw i64 %n, 1
  %exitcond17 = icmp ne i64 %inc28, %inc29
  br i1 %exitcond17, label %for.cond18.preheader, label %for.inc30.loopexit

for.inc30.loopexit:                               ; preds = %for.inc27
  %B.addr.3.lcssa.lcssa = phi ptr [ %B.addr.3.lcssa, %for.inc27 ]
  br label %for.inc30

for.inc30:                                        ; preds = %for.inc30.loopexit, %for.cond15.loopexit
  %B.addr.2.lcssa = phi ptr [ %B.addr.110, %for.cond15.loopexit ], [ %B.addr.3.lcssa.lcssa, %for.inc30.loopexit ]
  %inc31 = add nsw i64 %j.09, 1
  %exitcond18 = icmp ne i64 %inc31, %n
  br i1 %exitcond18, label %for.cond4.preheader, label %for.inc33.loopexit

for.inc33.loopexit:                               ; preds = %for.inc30
  %B.addr.2.lcssa.lcssa = phi ptr [ %B.addr.2.lcssa, %for.inc30 ]
  br label %for.inc33

for.inc33:                                        ; preds = %for.inc33.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi ptr [ %B.addr.014, %for.cond1.preheader ], [ %B.addr.2.lcssa.lcssa, %for.inc33.loopexit ]
  %inc34 = add nsw i64 %i.013, 1
  %exitcond19 = icmp ne i64 %inc34, %n
  br i1 %exitcond19, label %for.cond1.preheader, label %for.end35.loopexit

for.end35.loopexit:                               ; preds = %for.inc33
  br label %for.end35

for.end35:                                        ; preds = %for.end35.loopexit, %entry
  ret void
}
