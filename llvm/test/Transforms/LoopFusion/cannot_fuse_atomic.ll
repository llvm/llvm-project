; RUN: opt -S -passes=loop-simplify,loop-fusion \
; RUN:   -pass-remarks-analysis=loop-fusion -disable-output < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts

; Atomic accesses impose ordering constraints that the dependence analysis
; used by fusion does not model, so a loop containing one is not a fusion
; candidate.

; CHECK: [atomic_access]: Loop is not a candidate for fusion: Loop contains an atomic access

define void @atomic_access(ptr noalias %arg) {
entry:
  br label %for.body1

for.body1:                                        ; preds = %entry, %for.body1
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %for.body1 ]
  %gep1 = getelementptr inbounds i32, ptr %arg, i64 %i1
  %v1 = trunc i64 %i1 to i32
  store atomic i32 %v1, ptr %gep1 monotonic, align 4
  %inc1 = add nuw nsw i64 %i1, 1
  %cond1 = icmp ne i64 %inc1, 100
  br i1 %cond1, label %for.body1, label %for.body2.preheader

for.body2.preheader:                              ; preds = %for.body1
  br label %for.body2

for.body2:                                        ; preds = %for.body2.preheader, %for.body2
  %i2 = phi i64 [ 0, %for.body2.preheader ], [ %inc2, %for.body2 ]
  %gep2 = getelementptr inbounds i32, ptr %arg, i64 %i2
  %v2 = trunc i64 %i2 to i32
  store i32 %v2, ptr %gep2, align 4
  %inc2 = add nuw nsw i64 %i2, 1
  %cond2 = icmp ne i64 %inc2, 100
  br i1 %cond2, label %for.body2, label %exit

exit:                                             ; preds = %for.body2
  ret void
}
