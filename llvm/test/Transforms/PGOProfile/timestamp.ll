; RUN: opt < %s -passes=pgo-instr-gen -pgo-temporal-instrumentation -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-gen -pgo-temporal-instrumentation -pgo-block-coverage -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  ; CHECK: call void @llvm.instrprof.timestamp({{.*}})
  ret void
}

; CHECK: declare void @llvm.instrprof.timestamp(
