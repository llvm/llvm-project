; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary 1.ll -o 1.bc
; RUN: opt -module-summary 2.ll -o 2.bc
; RUN: llvm-lto -thinlto -o 3 1.bc 2.bc
; RUN: opt -S -passes=function-import -summary-file 3.thinlto.bc 1.bc 2>&1 | FileCheck %s

; CHECK: Function Import: link error: linking module flags 'Error': IDs have conflicting values in '2.bc' and '1.bc'

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  call void () @foo()
  ret i32 0
}

declare void @foo()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"Error", i32 0}

;--- 2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"Error", i32 1}
