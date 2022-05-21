; RUN: opt %loadPolly -polly-delicm -polly-simplify -polly-opt-isl \
; RUN: -polly-pattern-matching-based-opts=true \
; RUN: -polly-tc-opt=true -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Check that a region statement, which has the correct order of accesses, is not
; detected.
;
;    for (int i = 0; i < 32; i++)
;      for (int j = 0; j < 32; j++)
;        for (int k = 0; k < 32; k++) {
;          int c = C[i][j];
;          if (i*j*k < 10) {
;            C[i][j] = A[i][k] + B[k][j];
;          } else {
;            C[i][j] = c;
;          }
;        }
;
; CHECK-NOT: The tensor contraction pattern was detected
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo([64 x double]* noundef %C, [64 x double]* noundef %A, [64 x double]* noundef %B) #0 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc34
  %indvars.iv48 = phi i64 [ 0, %entry ], [ %indvars.iv.next49, %for.inc34 ]
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.inc31
  %indvars.iv43 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next44, %for.inc31 ]
  br label %for.body8

for.body8:                                        ; preds = %for.cond5.preheader, %if.end
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %if.end ]
  %arrayidx10 = getelementptr inbounds [64 x double], [64 x double]* %C, i64 %indvars.iv48, i64 %indvars.iv43
  %0 = mul nuw nsw i64 %indvars.iv43, %indvars.iv48
  %1 = mul nuw nsw i64 %0, %indvars.iv
  %cmp12 = icmp ult i64 %1, 10
  br i1 %cmp12, label %if.then, label %if.else

if.then:                                          ; preds = %for.body8
  %arrayidx17 = getelementptr inbounds [64 x double], [64 x double]* %A, i64 %indvars.iv48, i64 %indvars.iv
  %2 = load double, double* %arrayidx17, align 8
  %arrayidx21 = getelementptr inbounds [64 x double], [64 x double]* %B, i64 %indvars.iv, i64 %indvars.iv43
  %3 = load double, double* %arrayidx21, align 8
  %add = fadd fast double %3, %2
  br label %if.end

if.else:                                          ; preds = %for.body8
  %4 = load double, double* %arrayidx10, align 8
  %conv = fptosi double %4 to i32
  %conv26 = sitofp i32 %conv to double
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge = phi double [ %conv26, %if.else ], [ %add, %if.then ]
  store double %storemerge, double* %arrayidx10, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 32
  br i1 %exitcond, label %for.body8, label %for.inc31

for.inc31:                                        ; preds = %if.end
  %indvars.iv.next44 = add nuw nsw i64 %indvars.iv43, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next44, 32
  br i1 %exitcond47, label %for.cond5.preheader, label %for.inc34

for.inc34:                                        ; preds = %for.inc31
  %indvars.iv.next49 = add nuw nsw i64 %indvars.iv48, 1
  %exitcond51 = icmp ne i64 %indvars.iv.next49, 32
  br i1 %exitcond51, label %for.cond1.preheader, label %for.end36

for.end36:                                        ; preds = %for.inc34
  ret void
}
