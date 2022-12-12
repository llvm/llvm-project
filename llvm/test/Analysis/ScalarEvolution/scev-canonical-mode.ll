; PR26529: Check the assumption of IndVarSimplify to do SCEV expansion in literal mode
; instead of CanonicalMode is properly maintained in SCEVExpander::expand.
; RUN: opt -passes=indvars < %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @ehF() #0 {
entry:
  br i1 undef, label %if.then.i, label %hup.exit

if.then.i:                                        ; preds = %entry
  br i1 undef, label %for.body.lr.ph.i, label %hup.exit

for.body.lr.ph.i:                                 ; preds = %if.then.i
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  %i.03.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %for.body.i ]
  %k.02.i = phi i32 [ 1, %for.body.lr.ph.i ], [ %inc5.i, %for.body.i ]
  %inc.i = add nsw i32 %i.03.i, 1
  %idxprom.i = sext i32 %i.03.i to i64
  %idxprom2.i = sext i32 %k.02.i to i64
  %inc5.i = add nsw i32 %k.02.i, 1
  br i1 false, label %for.body.i, label %hup.exit

hup.exit:                                         ; preds = %for.body.i, %if.then.i, %entry
  ret void
}

attributes #0 = { norecurse nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
