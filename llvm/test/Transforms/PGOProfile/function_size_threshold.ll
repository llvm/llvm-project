; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --implicit-check-not='llvm.instrprof.' --check-prefixes=CHECK,ALL
; RUN: opt < %s -passes=pgo-instr-gen -pgo-function-size-threshold=2 -S | FileCheck %s --implicit-check-not='llvm.instrprof.' --check-prefixes=CHECK,ALL
; RUN: opt < %s -passes=pgo-instr-gen -pgo-function-size-threshold=3 -S | FileCheck %s --implicit-check-not='llvm.instrprof.'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define i32 @small(
define i32 @small(i32 %i) {
  ; ALL: call void @llvm.instrprof.increment({{.*}})
  %add = add i32 %i, 4
  ret i32 %add
}

; CHECK-LABEL: define i32 @large(
define i32 @large(i32 %0) {
  ; CHECK: call void @llvm.instrprof.increment({{.*}})
  %2 = shl nsw i32 %0, 3
  %3 = or i32 %2, 4
  %4 = mul i32 %3, %0
  %5 = or i32 %4, 3
  %6 = sdiv i32 2, %0
  %7 = add nsw i32 %5, %6
  %8 = mul nsw i32 %0, %0
  %9 = udiv i32 5, %8
  %10 = add nsw i32 %7, %9
  ret i32 %10
}

; CHECK: declare void @llvm.instrprof.increment({{.*}})
