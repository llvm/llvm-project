; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -S %s | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/206551
;
; Regression test for a verifier crash ("Instruction does not dominate all
; uses!") triggered when an escaping scalar (%cond47.us.us.us) is defined in a
; statement whose iteration domain is empty. The inner loop trip count depends
; on a parameter that the runtime check restricts to a range where the loop
; body never executes, making the defining statement's domain empty. Polly
; prunes it, but the scalar still escapes via the outer-loop header PHI.
;
; Without the fix the generated merge block refers to a definition that only
; exists in the original (unoptimized) copy, breaking dominance. With the fix
; the region stays versioned and a proper .merge PHI is built, so we check that
; the versioning infrastructure (.s2a alloca, .merge PHI, .final_reload) is
; present and well-formed.
;
; CHECK: %cond47.us.us.us.s2a = alloca
; CHECK: polly.merge_new_and_old:
; CHECK: %cond47.us.us.us.merge = phi i32 [ %cond47.us.us.us.final_reload, %polly.exiting ], [ %cond47.us.us.us, %for.cond.cleanup6.us.us ]
; CHECK: %cond47.us.us116.us = phi i32 [ 0, %entry ], [ %cond47.us.us.us.merge, %polly.merge_new_and_old ]
; CHECK: polly.exiting:
; CHECK: %cond47.us.us.us.final_reload = load i32, ptr %cond47.us.us.us.s2a

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_Z1iiPA4_A4_iiPA4_A4_cS4_S1_(ptr %j, i32 %k, ptr %0) {
entry:
  %sext = shl i32 %k, 24
  %conv10 = ashr i32 %sext, 24
  %sub11 = add i32 %conv10, -127
  %wide.trip.count = zext i32 %sub11 to i64
  br label %for.cond4.preheader.us.us

for.cond4.preheader.us.us:                        ; preds = %for.cond.cleanup6.us.us, %entry
  %cond47.us.us116.us = phi i32 [ 0, %entry ], [ %cond47.us.us.us, %for.cond.cleanup6.us.us ]
  br label %for.body7.us.us.us

for.cond.cleanup6.us.us:                          ; preds = %for.cond9.for.cond.cleanup13_crit_edge.us.us.us
  br label %for.cond4.preheader.us.us

for.body7.us.us.us:                               ; preds = %for.cond9.for.cond.cleanup13_crit_edge.us.us.us, %for.cond4.preheader.us.us
  br label %for.cond15.preheader.us.us.us

cond.false.us.us.us.1:                            ; preds = %for.cond15.preheader.us.us.us
  %cond47.us.us.us = select i1 false, i32 0, i32 0
  store i32 0, ptr null, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond9.for.cond.cleanup13_crit_edge.us.us.us, label %for.cond15.preheader.us.us.us

for.cond15.preheader.us.us.us:                    ; preds = %cond.false.us.us.us.1, %for.body7.us.us.us
  %indvars.iv = phi i64 [ %indvars.iv.next, %cond.false.us.us.us.1 ], [ 0, %for.body7.us.us.us ]
  %cond478486.us.us.us = phi i32 [ 1, %cond.false.us.us.us.1 ], [ 0, %for.body7.us.us.us ]
  %1 = load i32, ptr %0, align 4
  store i32 0, ptr %j, align 8
  br label %cond.false.us.us.us.1

for.cond9.for.cond.cleanup13_crit_edge.us.us.us:  ; preds = %cond.false.us.us.us.1
  br i1 true, label %for.cond.cleanup6.us.us, label %for.body7.us.us.us
}
