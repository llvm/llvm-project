; RUN: opt -ppc-vec-mask-cost=true -aa-pipeline=basic-aa -mcpu=pwr8 -S -passes=loop-vectorize < %s | FileCheck %s

target datalayout = "e-m:e-Fn32-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

define dso_local void @_tc(ptr nocapture noundef %aaa, i64 noundef %bbb) local_unnamed_addr {
; CHECK-NOT: extractelement <16 x i1>
entry:
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  ret void

for.body:                                         ; preds = %for.inc, %entry
  %i.08 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %aaa, i64 %i.08
  %0 = load i8, ptr %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  store i8 32, ptr %arrayidx, align 1
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %bbb
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
