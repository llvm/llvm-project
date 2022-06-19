; RUN: opt <%s -passes=pgo-instr-gen,instrprof -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i8* %dst, i8* %src, i32* %a, i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.cond1 ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.cond1, label %for.end6

for.cond1:
  %j.0 = phi i32 [ %inc, %for.body3 ], [ 0, %for.cond ]
  %idx.ext = sext i32 %i.0 to i64
  %add.ptr = getelementptr inbounds i32, i32* %a, i64 %idx.ext
  %0 = load i32, i32* %add.ptr, align 4
  %cmp2 = icmp slt i32 %j.0, %0
  %add = add nsw i32 %i.0, 1
  br i1 %cmp2, label %for.body3, label %for.cond

for.body3:
  %conv = sext i32 %add to i64
; CHECK: call void @__llvm_profile_instrument_memop(i64 %conv, i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_foo to i8*), i32 0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i1 false)
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end6:
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
