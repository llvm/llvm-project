; RUN: opt < %s -passes=pgo-instr-gen -pgo-function-entry-coverage -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %i) {
entry:
  ; CHECK: call void @llvm.instrprof.cover({{.*}})
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ; CHECK-NOT: llvm.instrprof.cover(
  %add = add nsw i32 %i, 2
  %s = select i1 %cmp, i32 %add, i32 0
  br label %if.end

if.else:
  %sub = sub nsw i32 %i, 2
  br label %if.end

if.end:
  %retv = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %retv
}

; CHECK: declare void @llvm.instrprof.cover(
