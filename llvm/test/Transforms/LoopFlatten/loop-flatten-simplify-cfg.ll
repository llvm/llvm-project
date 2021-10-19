;; This test was created to ensure that the LoopFlatten pass can
;; operate on loops that are not in simplified form before applying the pass.

; RUN: opt < %s -S -loop-flatten  -simplifycfg -verify-loop-info -verify-dom-info -verify-scev -verify | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @test(i64 %N, i64* %A, i64 %val) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i64 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_COND_PREHEADER_PREHEADER:%.*]]
; CHECK:       for.cond.preheader.preheader:
; CHECK-NEXT:    [[FLATTEN_TRIPCOUNT:%.*]] = mul i64 [[N]], [[N]]
; CHECK-NEXT:    br label [[FOR_COND_PREHEADER:%.*]]
; CHECK:       for.cond.preheader:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ [[INC6:%.*]], [[FOR_COND_PREHEADER]] ], [ 0, [[FOR_COND_PREHEADER_PREHEADER]] ]
; CHECK-NEXT:    [[MUL:%.*]] = mul i64 [[I]], [[N]]
; CHECK-NEXT:    [[ADD:%.*]] = add i64 0, [[MUL]]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i64, i64* [[A:%.*]], i64 [[I]]
; CHECK-NEXT:    [[TMP0:%.*]] = load i64, i64* [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[ADD4:%.*]] = add nsw i64 [[TMP0]], [[VAL:%.*]]
; CHECK-NEXT:    store i64 [[ADD4]], i64* [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[INC:%.*]] = add nuw nsw i64 0, 1
; CHECK-NEXT:    [[CMP2:%.*]] = icmp ult i64 [[INC]], [[N]]
; CHECK-NEXT:    [[INC6]] = add nuw nsw i64 [[I]], 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i64 [[INC6]], [[FLATTEN_TRIPCOUNT]]
; CHECK-NEXT:    br i1 [[CMP]], label [[FOR_COND_PREHEADER]], label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  %cmp.not = icmp eq i64 %N, 0
  br i1 %cmp.not, label %for.end, label %for.cond.preheader

for.cond.preheader:
  %i= phi i64 [ %inc6, %for.cond.for.inc_crit_edge ], [ 0, %entry ]
  %mul = mul i64 %i, %N
  br label %for.body

for.body:
  %j = phi i64 [ 0, %for.cond.preheader ], [ %inc, %for.body ]
  %add = add i64 %j, %mul
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  %0 = load i64, i64* %arrayidx, align 4
  %add4 = add nsw i64 %0, %val
  store i64 %add4, i64* %arrayidx, align 4
  %inc = add nuw nsw i64 %j, 1
  %cmp2 = icmp ult i64 %inc, %N
  br i1 %cmp2, label %for.body, label %for.cond.for.inc_crit_edge

for.cond.for.inc_crit_edge:
  %inc6 = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc6, %N
  br i1 %cmp, label %for.cond.preheader, label %for.end

for.end:
  ret void
}
