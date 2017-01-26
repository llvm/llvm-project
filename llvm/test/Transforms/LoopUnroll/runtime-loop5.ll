; RUN: opt < %s -S -loop-unroll -unroll-runtime=true -unroll-count=16 | FileCheck --check-prefix=UNROLL-16 %s
; RUN: opt < %s -S -loop-unroll -unroll-runtime=true -unroll-count=4 | FileCheck --check-prefix=UNROLL-4 %s

; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(unroll)' -unroll-runtime=true -unroll-count=16 | FileCheck --check-prefix=UNROLL-16 %s
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(unroll)' -unroll-runtime=true -unroll-count=4 | FileCheck --check-prefix=UNROLL-4 %s

; Given that the trip-count of this loop is a 3-bit value, we cannot
; safely unroll it with a count of anything more than 8.

define i3 @test(i3* %a, i3 %n) {
; UNROLL-16-LABEL: @test(
; UNROLL-4-LABEL: @test(
entry:
  %cmp1 = icmp eq i3 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

; UNROLL-16-NOT: for.body.prol:
; UNROLL-4: for.body.prol:

for.body:                                         ; preds = %for.body, %entry
; UNROLL-16-LABEL: for.body:
; UNROLL-4-LABEL: for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i3 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i3, i3* %a, i64 %indvars.iv

; UNROLL-16-LABEL: for.body
; UNROLL-16-LABEL: getelementptr
; UNROLL-16-LABEL-NOT: getelementptr

; UNROLL-4-LABEL: getelementptr
; UNROLL-4-LABEL: getelementptr
; UNROLL-4-LABEL: getelementptr
; UNROLL-4-LABEL: getelementptr

  %0 = load i3, i3* %arrayidx
  %add = add nsw i3 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i3
  %exitcond = icmp eq i3 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

; UNROLL-16-LABEL: for.end
; UNROLL-4-LABEL: for.end
for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i3 [ 0, %entry ], [ %add, %for.body ]
  ret i3 %sum.0.lcssa
}
