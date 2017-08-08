; RUN: llvm-profdata merge %S/Inputs/memop_size_annotation.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -memop-max-annotations=9 -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefixes=MEMOP_ANNOTATION,MEMOP_ANNOTATION9
; RUN: opt < %s -passes=pgo-instr-use -memop-max-annotations=9 -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefixes=MEMOP_ANNOTATION,MEMOP_ANNOTATION9
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefixes=MEMOP_ANNOTATION,MEMOP_ANNOTATION4
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefixes=MEMOP_ANNOTATION,MEMOP_ANNOTATION4

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i8* %dst, i8* %src, i32* %a, i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc4 ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end6

for.body:
  br label %for.cond1

for.cond1:
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %idx.ext = sext i32 %i.0 to i64
  %add.ptr = getelementptr inbounds i32, i32* %a, i64 %idx.ext
  %0 = load i32, i32* %add.ptr, align 4
  %cmp2 = icmp slt i32 %j.0, %0
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:
  %add = add nsw i32 %i.0, 1
  %conv = sext i32 %add to i64
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i32 1, i1 false)
; MEMOP_ANNOTATION: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i32 1, i1 false)
; MEMOP_ANNOTATION-SAME: !prof ![[MEMOP_VALUESITE:[0-9]+]]
; MEMOP_ANNOTATION9: ![[MEMOP_VALUESITE]] = !{!"VP", i32 1, i64 556, i64 1, i64 99, i64 2, i64 88, i64 3, i64 77, i64 9, i64 72, i64 4, i64 66, i64 5, i64 55, i64 6, i64 44, i64 7, i64 33, i64 8, i64 22}
; MEMOP_ANNOTATION4: ![[MEMOP_VALUESITE]] = !{!"VP", i32 1, i64 556, i64 1, i64 99, i64 2, i64 88, i64 3, i64 77, i64 9, i64 72}
  br label %for.inc

for.inc:
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:
  br label %for.inc4

for.inc4:
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end6:
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)

declare void @llvm.lifetime.end(i64, i8* nocapture)
