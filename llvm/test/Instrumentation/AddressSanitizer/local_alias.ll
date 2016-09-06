; RUN: opt < %s -asan -asan-module -asan-use-private-alias=1 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = internal global [2 x i32] zeroinitializer, align 4

; Check that we generate internal alias and odr indicator symbols for global to be protected.
; CHECK: @__odr_asan_gen_a = internal global i8 0, align 1
; CHECK: @"a<stdin>" = internal alias { [2 x i32], [56 x i8] }, { [2 x i32], [56 x i8] }* @a

; Function Attrs: nounwind sanitize_address uwtable
define i32 @foo(i32 %M) #0 {
entry:
  %M.addr = alloca i32, align 4
  store i32 %M, i32* %M.addr, align 4
  store volatile i32 6, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @a, i64 2, i64 0), align 4
  %0 = load i32, i32* %M.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* @a, i64 0, i64 %idxprom
  %1 = load volatile i32, i32* %arrayidx, align 4
  ret i32 %1
}
