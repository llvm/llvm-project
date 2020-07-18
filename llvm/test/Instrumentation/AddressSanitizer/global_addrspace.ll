; Only verify that asan don't crash on global variables of different
; address space. The global variable should be unmodified by asan.

; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = internal addrspace(42) global [1 x i32] zeroinitializer, align 4

; CHECK: @a = internal addrspace(42) global [1 x i32] zeroinitializer, align 4

define void @b(i32 %c) {
entry:
  %conv = sext i32 %c to i64
  %0 = inttoptr i64 %conv to i32 addrspace(42)*
  %cmp = icmp ugt i32 addrspace(42)* %0, getelementptr inbounds ([1 x i32], [1 x i32] addrspace(42)* @a, i64 0, i64 0)
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %call = tail call i32 (...) @e()
  br label %if.end

if.end:
  ret void
}

declare i32 @e(...)

!llvm.asan.globals = !{!0}
!0 = !{[1 x i32] addrspace(42)* @a, null, !"a", i1 false, i1 false}
