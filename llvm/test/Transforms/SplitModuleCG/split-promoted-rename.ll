; Test that internal symbols promoted during module splitting are consistently
; renamed with an MD5 suffix across all partitions.
;
; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -o %t \
; RUN:   -thinlto-split=true \
; RUN:   -thinlto-split-partitions=2 -thinlto-split-module-size-threshold=0 \
; RUN:   -r=%t.bc,caller_a,px \
; RUN:   -r=%t.bc,caller_b,px
; RUN: llvm-nm %t.1 | FileCheck %s

; CHECK-DAG: T caller_a
; CHECK-DAG: T caller_b
; CHECK:     T {{.*promoted_internal[._][0-9a-f]+.*}}
; CHECK-NOT: T promoted_internal{{$}}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; @promoted_internal is internal. SplitModuleCG::dealWithMpart's checkPromoted
; records it in PromotedRenames. splitOptAndCodeGenThin applies the rename
; after opt via:
;   for (auto &GV : MPart->global_values())
;     if (auto It = PromotedRenames.find(GV.getName()); ...)
;       GV.setName(It->second);
define internal void @promoted_internal() {
entry:
  ret void
}

define void @caller_a() {
entry:
  call void @promoted_internal()
  ret void
}

define void @caller_b() {
entry:
  call void @promoted_internal()
  ret void
}
