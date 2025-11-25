; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -enable-arm-maskedgatscat -tail-predication=force-enabled -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-COST,CHECK-COST-2
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-none-none-eabi"

define void @pred_loop(ptr %off, ptr %data, ptr %dst, i32 %n) #0 {

; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %i.09 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
; CHECK-COST-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %add = add nuw nsw i32 %i.09, 1
; CHECK-COST-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx = getelementptr inbounds i32, ptr %data, i32 %add
; CHECK-COST-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i32, ptr %arrayidx, align 4
; CHECK-COST-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %add1 = add nsw i32 %0, 5
; CHECK-COST-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx2 = getelementptr inbounds i32, ptr %dst, i32 %i.09
; CHECK-COST-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %add1, ptr %arrayidx2, align 4
; CHECK-COST-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %exitcond.not = icmp eq i32 %add, %n
; CHECK-COST-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %exitcond.not, label %exit.loopexit, label %for.body
; CHECK-COST-NEXT: LV: Scalar loop costs: 5.

entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body, label %exit

exit:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
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

; CHECK-COST-2: LV: Found an estimated cost of 0 for VF 1 For instruction:   %i.032 = phi i32 [ %inc, %if.end ], [ %start, %for.body.preheader ]
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.032
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i32, ptr %arrayidx, align 4
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx2 = getelementptr inbounds i32, ptr %b, i32 %i.032
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i32, ptr %arrayidx2, align 4
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp3 = icmp sgt i32 %0, %1
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp3, label %if.then, label %if.end
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nsw i32 %0, 5
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %add = add nsw i32 %mul, 3
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %factor = shl i32 %add, 1
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %sub = sub i32 %0, %1
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %add7 = add i32 %sub, %factor
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %add7, ptr %arrayidx2, align 4
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br label %if.end
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %k.0 = phi i32 [ %add, %if.then ], [ %0, %for.body ]
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %k.0, ptr %arrayidx, align 4
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %inc = add nsw i32 %i.032, 1
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %exitcond.not = icmp eq i32 %inc, %end
; CHECK-COST-2-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
; CHECK-COST-2-NEXT: LV: Scalar loop costs: 8.

entry:
  %cmp31 = icmp slt i32 %start, %end
  br i1 %cmp31, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %if.end
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %if.end
  %i.032 = phi i32 [ %inc, %if.end ], [ %start, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.032
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i32 %i.032
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %mul = mul nsw i32 %0, 5
  %add = add nsw i32 %mul, 3
  %factor = shl i32 %add, 1
  %sub = sub i32 %0, %1
  %add7 = add i32 %sub, %factor
  store i32 %add7, ptr %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %k.0 = phi i32 [ %add, %if.then ], [ %0, %for.body ]
  store i32 %k.0, ptr %arrayidx, align 4
  %inc = add nsw i32 %i.032, 1
  %exitcond.not = icmp eq i32 %inc, %end
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

attributes #0 = { "target-features"="+armv8.1-m.main,+dsp,+fp-armv8d16sp,+fp16,+fullfp16,+hwdiv,+lob,+mve,+mve.fp,+ras,+strict-align,+thumb-mode,+vfp2sp,+vfp3d16sp,+vfp4d16sp"}
