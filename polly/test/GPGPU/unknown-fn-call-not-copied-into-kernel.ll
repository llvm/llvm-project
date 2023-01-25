; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s --check-prefix=SCOP
; RUN: opt %loadPolly -S -polly-codegen-ppcg < %s | FileCheck %s

; Check that we do not create a kernel if there is an
; unknown function call in a candidate kernel.

; Check that we model the kernel as a scop.
; SCOP:      Function: f
; SCOP-NEXT:     Region: %entry.split---%for.end13

; If a kernel were generated, then this code would have been part of the kernel
; and not the `.ll` file that is generated.
; CHECK:       %conv = fpext float %0 to double
; CHECK-NEXT:  %1 = tail call double @extern.fn(double %conv)
; CHECK-NEXT:  %conv6 = fptrunc double %1 to float

; REQUIRES: pollyacc

; static const int N = 1000;
; void f(float A[N][N], int n, float B[N][N]) {
;   for(int i = 0; i < n; i++) {
;     for(int j = 0; j < n; j++) {
;       B[i][j] = extern_fn(A[i][j], 3);
;     }
;
;   }
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @f(ptr %A, i32 %n, ptr %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp3 = icmp sgt i32 %n, 0
  br i1 %cmp3, label %for.cond1.preheader.lr.ph, label %for.end13

for.cond1.preheader.lr.ph:                        ; preds = %entry.split
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.inc11
  %indvars.iv5 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next6, %for.inc11 ]
  %cmp21 = icmp sgt i32 %n, 0
  br i1 %cmp21, label %for.body3.lr.ph, label %for.inc11

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [1000 x float], ptr %A, i64 %indvars.iv5, i64 %indvars.iv
  %0 = load float, ptr %arrayidx5, align 4
  %conv = fpext float %0 to double
  %1 = tail call double @extern.fn(double %conv)
  %conv6 = fptrunc double %1 to float
  %arrayidx10 = getelementptr inbounds [1000 x float], ptr %B, i64 %indvars.iv5, i64 %indvars.iv
  store float %conv6, ptr %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %wide.trip.count = zext i32 %n to i64
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body3, label %for.cond1.for.inc11_crit_edge

for.cond1.for.inc11_crit_edge:                    ; preds = %for.body3
  br label %for.inc11

for.inc11:                                        ; preds = %for.cond1.for.inc11_crit_edge, %for.cond1.preheader
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %wide.trip.count7 = zext i32 %n to i64
  %exitcond8 = icmp ne i64 %indvars.iv.next6, %wide.trip.count7
  br i1 %exitcond8, label %for.cond1.preheader, label %for.cond.for.end13_crit_edge

for.cond.for.end13_crit_edge:                     ; preds = %for.inc11
  br label %for.end13

for.end13:                                        ; preds = %for.cond.for.end13_crit_edge, %entry.split
  ret void
}

declare double @extern.fn(double) #0
attributes #0 = { readnone }
