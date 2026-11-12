;; Check that only one data variable is created when an instrprof.increment is
;; inlined into more than one function.
; RUN: opt %s -passes='cgscc(inline),instrprof' -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__profd_foobar = private global
; CHECK-NOT @__profd_foobar

declare void @llvm.instrprof.increment(ptr %0, i64 %1, i32 %2, i32 %3)
@__profn_foobar = private constant [6 x i8] c"foobar"

define internal void @foobar() {
  call void @llvm.instrprof.increment(ptr @__profn_foobar, i64 123456, i32 32, i32 0)
  ret void
}

define void @foo() {
  call void @foobar()
  ret void
}

define void @bar() {
  call void @foobar()
  ret void
}
