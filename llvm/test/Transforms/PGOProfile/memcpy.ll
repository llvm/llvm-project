; RUN: opt <%s -passes=pgo-instr-gen,instrprof -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %dst, ptr %src, ptr %a, i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.cond1 ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.cond1, label %for.end6

for.cond1:
  %j.0 = phi i32 [ %inc, %for.body3 ], [ 0, %for.cond ]
  %idx.ext = sext i32 %i.0 to i64
  %add.ptr = getelementptr inbounds i32, ptr %a, i64 %idx.ext
  %0 = load i32, ptr %add.ptr, align 4
  %cmp2 = icmp slt i32 %j.0, %0
  %add = add nsw i32 %i.0, 1
  br i1 %cmp2, label %for.body3, label %for.cond

for.body3:
  %conv = sext i32 %add to i64
; CHECK: call void @__llvm_profile_instrument_memop(i64 %conv, ptr @__profd_foo, i32 0)
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %conv, i1 false)
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end6:
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
