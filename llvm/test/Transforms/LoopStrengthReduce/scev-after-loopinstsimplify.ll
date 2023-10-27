; RUN: opt %s -passes=indvars,loop-instsimplify,loop-reduce
; We are only checking that there is no crash!

; https://bugs.llvm.org/show_bug.cgi?id=37936

; The problem is as follows:
; 1. indvars marks %dec as NUW.
; 2. loop-instsimplify runs instsimplify, which constant-folds %dec to -1
; 3. loop-reduce tries to do some further modification, but crashes
;    with an type assertion in cast, because %dec is no longer an Instruction,
;    even though the SCEV data indicated it was.

; If the runline is split into two, i.e. -indvars -loop-instsimplify first, that
; stored into a file, and then -loop-reduce is run on that, there is no crash.
; So it looks like the problem is due to -loop-instsimplify not discarding SCEV.

target datalayout = "n16"

@a = external global i16, align 1

define void @f1() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %land.end, %entry
  %c.0 = phi i16 [ 0, %entry ], [ %dec, %land.end ]
  br i1 undef, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %0 = load i16, ptr @a, align 1
  %cmp = icmp sgt i16 %0, %c.0
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %for.body
  unreachable

land.end:                                         ; preds = %for.body
  %dec = add nsw i16 %c.0, -1
  br label %for.cond
}
