; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-arm-none-eabi"

; CHECK-LABEL: t1
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k] =
define void @t1(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [0 0 0|<]!
; CHECK: da analyze - none!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  store i32 %add12, ptr %arrayidx, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t2
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k + 1] =
define void @t2(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - anti [* * *|<]!
; CHECK: da analyze - output [* * *]!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %add111 = add nsw i32 %add11, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t3
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k - 1] =
define void @t3(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - anti [* * *|<]!
; CHECK: da analyze - output [* * *]!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %add111 = sub nsw i32 %add11, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t4
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k + o] =
define void @t4(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - anti [* * *|<]!
; CHECK: da analyze - output [* * *]!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %add111 = add nsw i32 %add11, %o
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t5
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k - o] =
define void @t5(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - anti [* * *|<]!
; CHECK: da analyze - output [* * *]!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %add111 = sub nsw i32 %add11, %o
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t6
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k + m*o] =
define void @t6(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [-1 0 0]!
; CHECK: da analyze - none!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %mo = mul i32 %m, %o
  %add111 = add nsw i32 %add11, %mo
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t7
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 0; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k - m*o] =
define void @t7(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [1 0 0]!
; CHECK: da analyze - none!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %mo = mul i32 %m, %o
  %add111 = sub nsw i32 %add11, %mo
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: t8
;;  for (int i = 0; i < n; i++)
;;   for (int j = 0; j < m; j++)
;;    for (int k = 1; k < o; k++)
;;      = A[i*m*o + j*o + k]
;;     A[i*m*o + j*o + k - 1] =
define void @t8(i32 %n, i32 %m, i32 %o, ptr nocapture %A) {
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [0 0 1]!
; CHECK: da analyze - none!
entry:
  %cmp49 = icmp sgt i32 %n, 0
  br i1 %cmp49, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp247 = icmp sgt i32 %m, 0
  %cmp645 = icmp sgt i32 %o, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %i.050 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc23, %for.cond.cleanup3 ]
  br i1 %cmp247, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %i.050, %m
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.048 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc20, %for.cond.cleanup7 ]
  br i1 %cmp645, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %mul944 = add i32 %j.048, %mul
  %add = mul i32 %mul944, %o
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.046 = phi i32 [ 1, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add11 = add nsw i32 %k.046, %add
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add11
  %0 = load i32, ptr %arrayidx, align 4
  %add12 = add nsw i32 %0, 1
  %add111 = sub nsw i32 %add11, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add111
  store i32 %add12, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %k.046, 1
  %exitcond = icmp eq i32 %inc, %o
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %inc20 = add nuw nsw i32 %j.048, 1
  %exitcond51 = icmp eq i32 %inc20, %m
  br i1 %exitcond51, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc23 = add nuw nsw i32 %i.050, 1
  %exitcond52 = icmp eq i32 %inc23, %n
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}


; CHECK-LABEL: test_sizes
define double @test_sizes(i16 %h, i16 %N, ptr nocapture %array) {
; CHECK: da analyze - consistent input [0 S]!
; CHECK: da analyze - anti [* *|<]!
; CHECK: da analyze - output [* *]!
entry:
  %cmp28 = icmp sgt i16 %N, 1
  br i1 %cmp28, label %for.body.lr.ph, label %for.end12

for.body.lr.ph:                                   ; preds = %entry
  %cmp425 = icmp slt i16 %h, 0
  %0 = add i16 %h, 1
  %wide.trip.count = zext i16 %N to i32
  br label %for.body

for.body:                                         ; preds = %for.inc10, %for.body.lr.ph
  %indvars.iv32 = phi i32 [ 1, %for.body.lr.ph ], [ %indvars.iv.next33, %for.inc10 ]
  %indvars.iv = phi i16 [ 2, %for.body.lr.ph ], [ %indvars.iv.next, %for.inc10 ]
  br i1 %cmp425, label %for.inc10, label %for.body5.lr.ph

for.body5.lr.ph:                                  ; preds = %for.body
  %1 = sext i16 %indvars.iv to i32
  %arrayidx = getelementptr inbounds i16, ptr %array, i32 %indvars.iv32
  br label %for.body5

for.body5:                                        ; preds = %for.body5, %for.body5.lr.ph
  %indvars.iv30 = phi i32 [ %indvars.iv.next31, %for.body5 ], [ %1, %for.body5.lr.ph ]
  %j.027 = phi i16 [ %inc, %for.body5 ], [ 0, %for.body5.lr.ph ]
  %2 = load i16, ptr %arrayidx, align 4
  %add6 = add nsw i16 %2, %j.027
  %arrayidx8 = getelementptr inbounds i16, ptr %array, i32 %indvars.iv30
  store i16 %add6, ptr %arrayidx8, align 4
  %inc = add nuw nsw i16 %j.027, 1
  %indvars.iv.next31 = add nsw i32 %indvars.iv30, 1
  %exitcond = icmp eq i16 %inc, %0
  br i1 %exitcond, label %for.inc10, label %for.body5

for.inc10:                                        ; preds = %for.body5, %for.body
  %indvars.iv.next33 = add nuw nsw i32 %indvars.iv32, 1
  %indvars.iv.next = add i16 %indvars.iv, %0
  %exitcond34 = icmp eq i32 %indvars.iv.next33, %wide.trip.count
  br i1 %exitcond34, label %for.end12, label %for.body

for.end12:                                        ; preds = %for.inc10, %entry
  ret double undef
}


; CHECK-LABEL: nonnegative
define void @nonnegative(ptr nocapture %A, i32 %N) {
; CHECK: da analyze - none!
; CHECK: da analyze - consistent output [0 0|<]!
; CHECK: da analyze - none!
entry:
  %cmp44 = icmp eq i32 %N, 0
  br i1 %cmp44, label %exit, label %for.outer

for.outer:
  %h.045 = phi i32 [ %add19, %for.latch ], [ 0, %entry ]
  %mul = mul i32 %h.045, %N
  br label %for.inner

for.inner:
  %i.043 = phi i32 [ 0, %for.outer ], [ %add16, %for.inner ]
  %add = add i32 %i.043, %mul
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add
  store i32 1, ptr %arrayidx, align 4
  store i32 2, ptr %arrayidx, align 4
  %add16 = add nuw i32 %i.043, 1
  %exitcond46 = icmp eq i32 %add16, %N
  br i1 %exitcond46, label %for.latch, label %for.inner

for.latch:
  %add19 = add nuw i32 %h.045, 1
  %exitcond47 = icmp eq i32 %add19, %N
  br i1 %exitcond47, label %exit, label %for.outer

exit:
  ret void
}
