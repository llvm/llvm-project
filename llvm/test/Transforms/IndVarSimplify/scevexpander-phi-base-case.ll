; RUN: opt -indvars -S < %s | FileCheck %s

define i32 @fn() {
entry:
  ret i32 10
}

define i32 @test_nested2(i32 %tnr) {
; CHECK-LABEL: test_nested2
; CHECK-NOT: %indvars.iv
; CHECK: %i.0

; indvars should not replace the i.0 variable with a new one; SCEVExpander
; should determine that the old one is good to reuse.

entry:
  %res = alloca i32, align 4
  store volatile i32 0, i32* %res, align 4
  %call = call i32 @fn()
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %cmp = icmp slt i32 %i.0, %call
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end8

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc5, %for.inc ]
  %cmp2 = icmp slt i32 %j.0, %i.0
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.cond1
  br label %for.end

for.body4:                                        ; preds = %for.cond1
  %0 = load volatile i32, i32* %res, align 4
  %inc = add nsw i32 %0, 1
  store volatile i32 %inc, i32* %res, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %inc5 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond.cleanup3
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %inc7 = add nsw i32 %i.0, 1
  br label %for.cond

for.end8:                                         ; preds = %for.cond.cleanup
  %1 = load volatile i32, i32* %res, align 4
  %cmp9 = icmp eq i32 %1, 45
  %conv = zext i1 %cmp9 to i32
  ret i32 %conv
}
