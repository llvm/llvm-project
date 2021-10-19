; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/pseudo-probe-desc-import.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc

; Don't import pseudo probe desc.
; RUN: llvm-lto -thinlto-action=import %t1.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s
; CHECK-NOT: {i64 6699318081062747564, i64 4294967295, !"foo"
; CHECK: !{i64 -2624081020897602054, i64 281479271677951, !"main"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  call void (...) @foo()
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1)
  ret i32 0
}

declare void @foo(...)

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }

!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 -2624081020897602054, i64 281479271677951, !"main", null}
