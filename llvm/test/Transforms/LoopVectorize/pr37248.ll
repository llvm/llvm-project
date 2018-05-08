; RUN: opt -passes='loop-vectorize' -force-vector-width=2 -S < %s | FileCheck %s
;
; Forcing VF=2 to trigger vector code gen
;
; This is a test case that let's vectorizer's code gen to generate
; more than one BasicBlocks in the loop body (emulated masked scatter)
; for those targets that do not support masked scatter. Broadcast
; code generation was previously dependent on loop body being
; a single basic block and this test case exposed incorrect code gen
; resulting in an assert in IL verification. Test passes if IL verification
; does not fail.
;
; Performing minimal check in the output to ensure the loop is actually
; vectorized.
;
; CHECK: vector.body

@a = external global [2 x i16], align 1

define void @f1() {
entry:
  br label %for.body

for.body:                                         ; preds = %land.end, %entry
  %0 = phi i32 [ undef, %entry ], [ %dec, %land.end ]
  br i1 undef, label %land.end, label %land.rhs

land.rhs:                                         ; preds = %for.body
  %1 = load i32, i32* undef, align 1
  br label %land.end

land.end:                                         ; preds = %land.rhs, %for.body
  %2 = trunc i32 %0 to i16
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* @a, i16 0, i16 %2
  store i16 undef, i16* %arrayidx, align 1
  %dec = add nsw i32 %0, -1
  %cmp = icmp sgt i32 %0, 1
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %land.end
  unreachable
}
