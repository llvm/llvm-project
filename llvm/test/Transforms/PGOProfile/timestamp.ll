; RUN: opt < %s -passes=pgo-instr-gen -pgo-temporal-instrumentation -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-gen -pgo-temporal-instrumentation -pgo-block-coverage -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define void @foo(
define void @foo() {
entry:
  ; CHECK: call void @llvm.instrprof.timestamp({{.*}})
  ret void
}

; CHECK-LABEL: define void @bar(
define void @bar() #0 {
entry:
  ; CHECK-NOT: call void @llvm.instrprof.timestamp({{.*}})
  call void asm sideeffect "retq;", "~{dirflag},~{fpsr},~{flags}"()
  unreachable
}

; CHECK-LABEL: declare void @llvm.instrprof.timestamp(

attributes #0 = { naked }
