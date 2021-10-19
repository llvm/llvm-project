; Test that we do not internalize functions that appear in the CFI jump table in
; the full LTO object file; any such functions will be referenced by the jump
; table.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-lto2 run -o %t2 -r %t,f1,p -r %t,f2,p -r %t,_start,px %t -save-temps
; RUN: llvm-dis %t2.1.2.internalize.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define void @f1()
define void @f1() !type !0 {
  ret void
}

; CHECK: define internal void @f2()
define void @f2() !type !1 {
  ret void
}

define i1 @_start(i1 %i) {
  %1 = select i1 %i, void ()* @f1, void ()* @f2
  %2 = bitcast void ()* %1 to i8*
  %3 = call i1 @llvm.type.test(i8* %2, metadata !"typeid1")
  ret i1 %3
}

declare i1 @llvm.type.test(i8*, metadata)

!llvm.module.flags = !{!2}

!0 = !{i64 0, !"typeid1"}
!1 = !{i64 0, !"typeid2"}
!2 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
