; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/llvm-stats-import.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc

; RUN: llvm-lto -thinlto-action=import %t1.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-MAIN
; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-FOO

; CHECK-MAIN: !0 = !{!"main", i64 123}
; CHECK-MAIN-NOT: !{!"foo", i64 456}

; CHECK-FOO: !0 = !{!"foo", i64 456}
; CHECK-FOO-NOT: !{!"main", i64 123}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  call void (...) @foo()
  ret i32 0
}

declare void @foo(...)

attributes #0 = { inaccessiblememonly nounwind willreturn }

!llvm.stats = !{!0}
!0 = !{!"main", i64 123}
