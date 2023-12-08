; RUN: opt -module-summary -passes=pseudo-probe %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/pseudo-probe-desc-import.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc

; Don't import pseudo probe desc.
; RUN: llvm-lto -thinlto-action=import %t1.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s


; Warn that current module is not pseudo-probe instrumented.
; RUN: opt -module-summary %s -o %t3.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.index.bc %t3.bc %t2.bc
; RUN: llvm-lto -thinlto-action=import %t3.bc -thinlto-index=%t3.index.bc -o /dev/null 2>&1 | FileCheck %s  --check-prefix=WARN


; CHECK-NOT: {i64 6699318081062747564, i64 4294967295, !"foo"
; CHECK: !{i64 -2624081020897602054, i64 281479271677951, !"main"

; WARN: warning: Pseudo-probe ignored: source module '{{.*}}' is compiled with -fpseudo-probe-for-profiling while destination module '{{.*}}' is not

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  call void (...) @foo()
  ret i32 0
}

declare void @foo(...)

attributes #0 = { inaccessiblememonly nounwind willreturn }
