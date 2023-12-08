; RUN: llc < %s -mtriple=lanai | FileCheck %s
; RUN: llc < %s -mtriple=lanai -disable-lanai-mem-alu-combiner | \
; RUN:   FileCheck %s -check-prefix=CHECK-DIS

; CHECK-LABEL: sum,
; CHECK: ld [%r{{[0-9]+}}++], %r{{[0-9]+}}{{$}}
; CHECK-DIS-LABEL: sum,
; CHECK-DIS-NOT: ++],

define i32 @sum(i32* inreg nocapture readonly %data, i32 inreg %n) {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %sum_.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %sum_.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %sum_.07 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %data, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum_.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
