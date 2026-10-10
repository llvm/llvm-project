; RUN: opt %loadNPMPolly -plugin-arg=Polly,-polly-process-unprofitable=false '-passes=polly-custom<detect>' -plugin-arg=Polly,-polly-print-detect -disable-output < %s 2>&1 | FileCheck %s
;
; The four loops are valid regions on their own, and ScopDetection expands the
; first one by one loop at a time until the call to @ext stops it. Every valid
; region has to keep its detection context while that happens; ScopDetection
; asserts this when it has detected all regions.
;
;    void ext(void);
;
;    void f(int n, double *A, double *B, double *C, double *D) {
;      ext();
;      for (int i = 0; i < n; i++)
;        A[i] = 1.0;
;      for (int i = 0; i < n; i++)
;        B[i] = 2.0;
;      for (int i = 0; i < n; i++)
;        C[i] = 3.0;
;      for (int i = 0; i < n; i++)
;        D[i] = A[i] + B[i] + C[i];
;      ext();
;    }
;
; CHECK: Valid Region for Scop: for.body => for.cond.cleanup24

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32 %n, ptr %A, ptr %B, ptr %C, ptr %D) {
entry:
  tail call void @ext()
  %cmp55 = icmp sgt i32 %n, 0
  br i1 %cmp55, label %for.body.preheader, label %for.cond.cleanup24

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.body5.preheader:                              ; preds = %for.body
  %wide.trip.count67 = zext nneg i32 %n to i64
  br label %for.body5

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %indvars.iv
  store double 1.000000e+00, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.body5.preheader, label %for.body

for.body15.preheader:                             ; preds = %for.body5
  %wide.trip.count72 = zext nneg i32 %n to i64
  br label %for.body15

for.body5:                                        ; preds = %for.body5.preheader, %for.body5
  %indvars.iv64 = phi i64 [ 0, %for.body5.preheader ], [ %indvars.iv.next65, %for.body5 ]
  %arrayidx7 = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %indvars.iv64
  store double 2.000000e+00, ptr %arrayidx7, align 8
  %indvars.iv.next65 = add nuw nsw i64 %indvars.iv64, 1
  %exitcond68.not = icmp eq i64 %indvars.iv.next65, %wide.trip.count67
  br i1 %exitcond68.not, label %for.body15.preheader, label %for.body5

for.body25.preheader:                             ; preds = %for.body15
  %wide.trip.count77 = zext nneg i32 %n to i64
  br label %for.body25

for.body15:                                       ; preds = %for.body15.preheader, %for.body15
  %indvars.iv69 = phi i64 [ 0, %for.body15.preheader ], [ %indvars.iv.next70, %for.body15 ]
  %arrayidx17 = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %indvars.iv69
  store double 3.000000e+00, ptr %arrayidx17, align 8
  %indvars.iv.next70 = add nuw nsw i64 %indvars.iv69, 1
  %exitcond73.not = icmp eq i64 %indvars.iv.next70, %wide.trip.count72
  br i1 %exitcond73.not, label %for.body25.preheader, label %for.body15

for.cond.cleanup24:                               ; preds = %for.body25, %entry
  tail call void @ext()
  ret void

for.body25:                                       ; preds = %for.body25.preheader, %for.body25
  %indvars.iv74 = phi i64 [ 0, %for.body25.preheader ], [ %indvars.iv.next75, %for.body25 ]
  %arrayidx27 = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %indvars.iv74
  %0 = load double, ptr %arrayidx27, align 8
  %arrayidx29 = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %indvars.iv74
  %1 = load double, ptr %arrayidx29, align 8
  %add = fadd double %0, %1
  %arrayidx31 = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %indvars.iv74
  %2 = load double, ptr %arrayidx31, align 8
  %add32 = fadd double %add, %2
  %arrayidx34 = getelementptr inbounds nuw [8 x i8], ptr %D, i64 %indvars.iv74
  store double %add32, ptr %arrayidx34, align 8
  %indvars.iv.next75 = add nuw nsw i64 %indvars.iv74, 1
  %exitcond78.not = icmp eq i64 %indvars.iv.next75, %wide.trip.count77
  br i1 %exitcond78.not, label %for.cond.cleanup24, label %for.body25
}

declare void @ext()
