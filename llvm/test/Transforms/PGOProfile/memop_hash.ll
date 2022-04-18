; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)

define i64 @foo1(i8* %a, i8* %b, i32 %s) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 %s, i1 false);
  ret i64 0
}

define i64 @foo2(i8* %a, i8* %b, i32 %s) {
entry:
  ret i64 0
}

; The two hashes should not be equal as the existence of the memcpy should change the hash.
;
; CHECK: @foo1
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_foo1, i32 0, i32 0), i64 [[FOO1_HASH:[0-9]+]], i32 1, i32 0)
; CHECK: @foo2
; CHECK-NOT: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_foo2, i32 0, i32 0), i64 [[FOO1_HASH]], i32 1, i32 0)
