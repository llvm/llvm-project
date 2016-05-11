; RUN: opt %loadPolly -polly-detect -polly-process-unprofitable=false -analyze \
; RUN:     -polly-allow-nonaffine-loops < %s | FileCheck %s
;
; Even if we allow non-affine loops we can only model the outermost loop, all
; other loops are boxed in non-affine regions. However, the inner loops can be
; distributed as black-boxes, thus we will recognize the outer loop as profitable.
;
; CHECK:  Valid Region for Scop: for.cond => for.end.51
;
;    void f(int *A) {
;      for (int i = 0; i < 100; i++) {
;        // Non-affine
;        for (int j = 0; j < i * i; j++)
;          for (int k = 0; k < i; k++)
;            A[i]++;
;        // Non-affine
;        for (int j = 0; j < i * i; j++)
;          // Non-affine
;          for (int k = 0; k < j; k++)
;            A[i]++;
;        // Non-affine
;        if (A[i])
;          for (int j = 0; j < 100; j++)
;            for (int k = 0; k < j * j; k++)
;              A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.49, %entry
  %indvars.iv5 = phi i64 [ %indvars.iv.next6, %for.inc.49 ], [ 0, %entry ]
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.inc.49 ], [ 0, %entry ]
  %exitcond9 = icmp ne i64 %indvars.iv5, 100
  br i1 %exitcond9, label %for.body, label %for.end.51

for.body:                                         ; preds = %for.cond
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc.8, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc9, %for.inc.8 ]
  %tmp = mul nsw i64 %indvars.iv5, %indvars.iv5
  %tmp10 = sext i32 %j.0 to i64
  %cmp2 = icmp slt i64 %tmp10, %tmp
  br i1 %cmp2, label %for.body.3, label %for.end.10

for.body.3:                                       ; preds = %for.cond.1
  br label %for.cond.4

for.cond.4:                                       ; preds = %for.inc, %for.body.3
  %k.0 = phi i32 [ 0, %for.body.3 ], [ %inc7, %for.inc ]
  %exitcond = icmp ne i32 %k.0, %indvars.iv
  br i1 %exitcond, label %for.body.6, label %for.end

for.body.6:                                       ; preds = %for.cond.4
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv5
  %tmp11 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp11, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.6
  %inc7 = add nuw nsw i32 %k.0, 1
  br label %for.cond.4

for.end:                                          ; preds = %for.cond.4
  br label %for.inc.8

for.inc.8:                                        ; preds = %for.end
  %inc9 = add nuw nsw i32 %j.0, 1
  br label %for.cond.1

for.end.10:                                       ; preds = %for.cond.1
  br label %for.cond.12

for.cond.12:                                      ; preds = %for.inc.26, %for.end.10
  %indvars.iv1 = phi i32 [ %indvars.iv.next2, %for.inc.26 ], [ 0, %for.end.10 ]
  %tmp12 = mul nsw i64 %indvars.iv5, %indvars.iv5
  %tmp13 = sext i32 %indvars.iv1 to i64
  %cmp14 = icmp slt i64 %tmp13, %tmp12
  br i1 %cmp14, label %for.body.15, label %for.end.28

for.body.15:                                      ; preds = %for.cond.12
  br label %for.cond.17

for.cond.17:                                      ; preds = %for.inc.23, %for.body.15
  %k16.0 = phi i32 [ 0, %for.body.15 ], [ %inc24, %for.inc.23 ]
  %exitcond3 = icmp ne i32 %k16.0, %indvars.iv1
  br i1 %exitcond3, label %for.body.19, label %for.end.25

for.body.19:                                      ; preds = %for.cond.17
  %arrayidx21 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv5
  %tmp14 = load i32, i32* %arrayidx21, align 4
  %inc22 = add nsw i32 %tmp14, 1
  store i32 %inc22, i32* %arrayidx21, align 4
  br label %for.inc.23

for.inc.23:                                       ; preds = %for.body.19
  %inc24 = add nuw nsw i32 %k16.0, 1
  br label %for.cond.17

for.end.25:                                       ; preds = %for.cond.17
  br label %for.inc.26

for.inc.26:                                       ; preds = %for.end.25
  %indvars.iv.next2 = add nuw nsw i32 %indvars.iv1, 1
  br label %for.cond.12

for.end.28:                                       ; preds = %for.cond.12
  %arrayidx30 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv5
  %tmp15 = load i32, i32* %arrayidx30, align 4
  %tobool = icmp eq i32 %tmp15, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.end.28
  br label %for.cond.32

for.cond.32:                                      ; preds = %for.inc.46, %if.then
  %j31.0 = phi i32 [ 0, %if.then ], [ %inc47, %for.inc.46 ]
  %exitcond4 = icmp ne i32 %j31.0, 100
  br i1 %exitcond4, label %for.body.34, label %for.end.48

for.body.34:                                      ; preds = %for.cond.32
  br label %for.cond.36

for.cond.36:                                      ; preds = %for.inc.43, %for.body.34
  %k35.0 = phi i32 [ 0, %for.body.34 ], [ %inc44, %for.inc.43 ]
  %mul37 = mul nsw i32 %j31.0, %j31.0
  %cmp38 = icmp slt i32 %k35.0, %mul37
  br i1 %cmp38, label %for.body.39, label %for.end.45

for.body.39:                                      ; preds = %for.cond.36
  %arrayidx41 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv5
  %tmp16 = load i32, i32* %arrayidx41, align 4
  %inc42 = add nsw i32 %tmp16, 1
  store i32 %inc42, i32* %arrayidx41, align 4
  br label %for.inc.43

for.inc.43:                                       ; preds = %for.body.39
  %inc44 = add nuw nsw i32 %k35.0, 1
  br label %for.cond.36

for.end.45:                                       ; preds = %for.cond.36
  br label %for.inc.46

for.inc.46:                                       ; preds = %for.end.45
  %inc47 = add nuw nsw i32 %j31.0, 1
  br label %for.cond.32

for.end.48:                                       ; preds = %for.cond.32
  br label %if.end

if.end:                                           ; preds = %for.end.28, %for.end.48
  br label %for.inc.49

for.inc.49:                                       ; preds = %if.end
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  br label %for.cond

for.end.51:                                       ; preds = %for.cond
  ret void
}
