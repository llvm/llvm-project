; RUN: opt < %s  -opaque-pointers -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

; CHECK-DAG: Loop 'for.j' has cost = 201000000
; CHECK-DAG: Loop 'for.i' has cost = 102000000
; CHECK-DAG: Loop 'for.k' has cost = 90000

;; Test to make sure when we have multiple conflicting access patterns, the 
;; chosen loop configuration favours the majority of those accesses. 
;; For example this nest should be ordered as j-i-k.
;;  for (int i = 0; i < n; i++)
;;    for (int j = 0; j < n; j++)
;;      for (int k = 0; k < n; k++) {
;;        A[i][j][k] = 1;
;;        B[j][i][k] = 2;
;;        C[j][i][k] = 3;
;;      }                            

define void @foo(i32 noundef signext %n, ptr noalias noundef %A, ptr noalias noundef %B, ptr noalias noundef %C) {
entry:
  %0 = zext i32 %n to i64
  %1 = zext i32 %n to i64
  %2 = zext i32 %n to i64
  %3 = zext i32 %n to i64
  %4 = zext i32 %n to i64
  %5 = zext i32 %n to i64
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.i.preheader, label %for.end30

for.i.preheader:                               ; preds = %entry
  %wide.trip.count16 = zext i32 %n to i64
  br label %for.i

for.i:                                         ; preds = %for.i.preheader, %for.inc28
  %indvars.iv13 = phi i64 [ 0, %for.i.preheader ], [ %indvars.iv.next14, %for.inc28 ]
  %cmp23 = icmp sgt i32 %n, 0
  br i1 %cmp23, label %for.j.preheader, label %for.inc28

for.j.preheader:                              ; preds = %for.i
  %wide.trip.count11 = zext i32 %n to i64
  br label %for.j

for.j:                                        ; preds = %for.j.preheader, %for.inc25
  %indvars.iv8 = phi i64 [ 0, %for.j.preheader ], [ %indvars.iv.next9, %for.inc25 ]
  %cmp61 = icmp sgt i32 %n, 0
  br i1 %cmp61, label %for.k.preheader, label %for.inc25

for.k.preheader:                              ; preds = %for.j
  %wide.trip.count = zext i32 %n to i64
  br label %for.k

for.k:                                        ; preds = %for.k.preheader, %for.k
  %indvars.iv = phi i64 [ 0, %for.k.preheader ], [ %indvars.iv.next, %for.k ]
  %6 = mul nuw i64 %0, %1
  %7 = mul nsw i64 %6, %indvars.iv13
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %7
  %8 = mul nuw nsw i64 %indvars.iv8, %1
  %arrayidx10 = getelementptr inbounds i32, ptr %arrayidx, i64 %8
  %arrayidx12 = getelementptr inbounds i32, ptr %arrayidx10, i64 %indvars.iv
  store i32 1, ptr %arrayidx12, align 4
  %9 = mul nuw i64 %2, %3
  %10 = mul nsw i64 %9, %indvars.iv8
  %arrayidx14 = getelementptr inbounds i32, ptr %B, i64 %10
  %11 = mul nuw nsw i64 %indvars.iv13, %3
  %arrayidx16 = getelementptr inbounds i32, ptr %arrayidx14, i64 %11
  %arrayidx18 = getelementptr inbounds i32, ptr %arrayidx16, i64 %indvars.iv
  store i32 2, ptr %arrayidx18, align 4
  %12 = mul nuw i64 %4, %5
  %13 = mul nsw i64 %12, %indvars.iv8
  %arrayidx20 = getelementptr inbounds i32, ptr %C, i64 %13
  %14 = mul nuw nsw i64 %indvars.iv13, %5
  %arrayidx22 = getelementptr inbounds i32, ptr %arrayidx20, i64 %14
  %arrayidx24 = getelementptr inbounds i32, ptr %arrayidx22, i64 %indvars.iv
  store i32 3, ptr %arrayidx24, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.k, label %for.inc25.loopexit

for.inc25.loopexit:                               ; preds = %for.k
  br label %for.inc25

for.inc25:                                        ; preds = %for.inc25.loopexit, %for.j
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %exitcond12 = icmp ne i64 %indvars.iv.next9, %wide.trip.count11
  br i1 %exitcond12, label %for.j, label %for.inc28.loopexit

for.inc28.loopexit:                               ; preds = %for.inc25
  br label %for.inc28

for.inc28:                                        ; preds = %for.inc28.loopexit, %for.i
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 1
  %exitcond17 = icmp ne i64 %indvars.iv.next14, %wide.trip.count16
  br i1 %exitcond17, label %for.i, label %for.end30.loopexit

for.end30.loopexit:                               ; preds = %for.inc28
  br label %for.end30

for.end30:                                        ; preds = %for.end30.loopexit, %entry
  ret void
}
