; REQUIRES: asserts
; RUN: opt -passes=gvn -debug-only=local -disable-output 2>&1 < %s | FileCheck %s

define void @replaceDominatedUsesWith_debug(ptr nocapture writeonly %a, i32 %beam) {
; CHECK: Replace dominated use of 'i64 %indvars.iv' with   %0 = zext i32 %beam to i64 in   %1 = shl nuw nsw i64 %indvars.iv, 1
entry:
  %0 = zext i32 %beam to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %cmp1 = icmp eq i64 %indvars.iv, %0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %1 = shl nuw nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %1
  store i32 0, ptr %arrayidx, align 4
  br label %for.inc

if.else:                                          ; preds = %for.body
  %2 = shl nuw nsw i64 %indvars.iv, 1
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 1, ptr %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 10000
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
