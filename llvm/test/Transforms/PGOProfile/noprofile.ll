; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@i = dso_local global i32 0, align 4

define i32 @test1() {
entry:
; CHECK: call void @llvm.instrprof.increment
  %0 = load i32, ptr @i, align 4
  %add = add i32 %0, 1
  ret i32 %add
}

define i32 @test2() #0 {
entry:
; CHECK-NOT: call void @llvm.instrprof.increment
  %0 = load i32, ptr @i, align 4
  %sub = sub i32 %0, 1
  ret i32 %sub
}

attributes #0 = { noprofile }
