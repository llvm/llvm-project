; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -enable-arm-maskedgatscat -tail-predication=force-enabled -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-COST,CHECK-COST-2
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-none-none-eabi"

define void @pred_loop(ptr %off, ptr %data, ptr %dst, i32 %n) #0 {

; CHECK-COST-LABEL: LV: Checking a loop in 'pred_loop'
; CHECK-COST:      Cost of 0 for VF 1: EMIT-SCALAR ir<%i.09> = phi [ ir<0>, vector.ph ], [ ir<%add>, for.body ]
; CHECK-COST-NEXT: Cost of 1 for VF 1: EMIT ir<%add> = add nuw nsw ir<%i.09>, ir<1>
; CHECK-COST-NEXT: Cost of 0 for VF 1: EMIT ir<%arrayidx> = getelementptr inbounds ir<%data>, ir<%add>
; CHECK-COST-NEXT: Cost of 1 for VF 1: EMIT-SCALAR ir<%0> = load ir<%arrayidx>
; CHECK-COST-NEXT: Cost of 1 for VF 1: EMIT ir<%add1> = add nsw ir<%0>, ir<5>
; CHECK-COST-NEXT: Cost of 0 for VF 1: EMIT ir<%arrayidx2> = getelementptr inbounds ir<%dst>, ir<%i.09>
; CHECK-COST-NEXT: Cost of 1 for VF 1: EMIT store ir<%add1>, ir<%arrayidx2>
; CHECK-COST-NEXT: Cost of 1 for VF 1: EMIT ir<%exitcond.not> = icmp eq ir<%add>, ir<%n>
; CHECK-COST-NEXT: Cost of 0 for VF 1: EMIT branch-on-cond ir<%exitcond.not>
; CHECK-COST:      LV: Scalar loop costs: 5.

entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body, label %exit

exit:
  ret void

for.body:
  %i.09 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %add = add nuw nsw i32 %i.09, 1
  %arrayidx = getelementptr inbounds i32, ptr %data, i32 %add
  %0 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %0, 5
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i32 %i.09
  store i32 %add1, ptr %arrayidx2, align 4
  %exitcond.not = icmp eq i32 %add, %n
  br i1 %exitcond.not, label %exit, label %for.body
}

define void @if_convert(ptr %a, ptr %b, i32 %start, i32 %end) #0 {

; CHECK-COST-2-LABEL: LV: Checking a loop in 'if_convert'
; CHECK-COST-2:      Cost of 0 for VF 1: EMIT-SCALAR ir<%i.032> = phi [ ir<%start>, vector.ph ], [ ir<%inc>, if.end ]
; CHECK-COST-2-NEXT: Cost of 0 for VF 1: EMIT ir<%arrayidx> = getelementptr inbounds ir<%a>, ir<%i.032>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT-SCALAR ir<%0> = load ir<%arrayidx>
; CHECK-COST-2-NEXT: Cost of 0 for VF 1: EMIT ir<%arrayidx2> = getelementptr inbounds ir<%b>, ir<%i.032>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT-SCALAR ir<%1> = load ir<%arrayidx2>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%cmp3> = icmp sgt ir<%0>, ir<%1>
; CHECK-COST-2-NEXT: Cost of 0 for VF 1: EMIT branch-on-cond ir<%cmp3>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%mul> = mul nsw ir<%0>, ir<5>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%add> = add nsw ir<%mul>, ir<3>
; CHECK-COST-2-NEXT: Cost of 0 for VF 1: EMIT ir<%factor> = shl ir<%add>, ir<1>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%sub> = sub ir<%0>, ir<%1>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%add7> = add ir<%sub>, ir<%factor>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT store ir<%add7>, ir<%arrayidx2>
; CHECK-COST-2-NEXT: Cost of 0 for VF 1: EMIT-SCALAR ir<%k.0> = phi [ ir<%add>, if.then ], [ ir<%0>, for.body ]
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT store ir<%k.0>, ir<%arrayidx>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%inc> = add nsw ir<%i.032>, ir<1>
; CHECK-COST-2-NEXT: Cost of 1 for VF 1: EMIT ir<%exitcond.not> = icmp eq ir<%inc>, ir<%end>
; CHECK-COST-2-NEXT: Cost of 0 for VF 1: EMIT branch-on-cond ir<%exitcond.not>
; CHECK-COST-2:      LV: Scalar loop costs: 8.5.

entry:
  %cmp31 = icmp slt i32 %start, %end
  br i1 %cmp31, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.032 = phi i32 [ %inc, %if.end ], [ %start, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.032
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i32 %i.032
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  br i1 %cmp3, label %if.then, label %if.end

if.then:
  %mul = mul nsw i32 %0, 5
  %add = add nsw i32 %mul, 3
  %factor = shl i32 %add, 1
  %sub = sub i32 %0, %1
  %add7 = add i32 %sub, %factor
  store i32 %add7, ptr %arrayidx2, align 4
  br label %if.end

if.end:
  %k.0 = phi i32 [ %add, %if.then ], [ %0, %for.body ]
  store i32 %k.0, ptr %arrayidx, align 4
  %inc = add nsw i32 %i.032, 1
  %exitcond.not = icmp eq i32 %inc, %end
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

attributes #0 = { "target-features"="+armv8.1-m.main,+dsp,+fp-armv8d16sp,+fp16,+fullfp16,+hwdiv,+lob,+mve,+mve.fp,+ras,+strict-align,+thumb-mode,+vfp2sp,+vfp3d16sp,+vfp4d16sp"}
