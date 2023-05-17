; Check for recognizing the "memmove" idiom.
; RUN: opt -hexagon-loop-idiom -S -mtriple hexagon-unknown-elf < %s \
; RUN:  | FileCheck %s
; CHECK: call void @llvm.memmove

; Function Attrs: norecurse nounwind
define void @foo(ptr nocapture %A, ptr nocapture readonly %B, i32 %n) #0 {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %arrayidx.phi = phi ptr [ %B, %for.body.preheader ], [ %arrayidx.inc, %for.body ]
  %arrayidx1.phi = phi ptr [ %A, %for.body.preheader ], [ %arrayidx1.inc, %for.body ]
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %0 = load i32, ptr %arrayidx.phi, align 4
  store i32 %0, ptr %arrayidx1.phi, align 4
  %inc = add nuw nsw i32 %i.02, 1
  %exitcond = icmp ne i32 %inc, %n
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr i32, ptr %arrayidx1.phi, i32 1
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

attributes #0 = { nounwind }
