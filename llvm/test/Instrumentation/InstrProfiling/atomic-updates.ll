; RUN: opt < %s -S -passes=instrprof -instrprof-atomic-counter-update-all | FileCheck %s

target triple = "x86_64-apple-macosx10.10.0"

@__profn_foo = private constant [3 x i8] c"foo"

; CHECK-LABEL: define void @foo
; CHECK-NEXT: atomicrmw add ptr @__profc_foo, i64 1 monotonic
define void @foo() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
