; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses -debug-only=loop-vectorize,vectorutils -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

; Ensure that we don't create interleave groups for predicated
; strided accesses.

; CHECK: LV: Checking a loop in 'masked_strided'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NOT: LV: Creating an interleave group

define dso_local void @masked_strided(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.017 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.017, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.017, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx4 = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 %0, i8* %arrayidx4, align 1
  %sub = sub i8 0, %0
  %add = or i32 %mul, 1
  %arrayidx8 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 %sub, i8* %arrayidx8, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.017, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
