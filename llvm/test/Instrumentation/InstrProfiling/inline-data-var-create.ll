;; Check that all data variables are created for instrumented functions even
;; when those functions are fully inlined into their instrumented callers prior
;; to the instrprof pass.
; RUN: opt %s -passes='instrprof' -S | FileCheck %s -check-prefix=NOINLINE
; RUN: opt %s -passes='cgscc(inline),instrprof' -S | FileCheck %s -check-prefix=INLINEFIRST
; RUN: opt %s -passes='instrprof,cgscc(inline)' -S | FileCheck %s -check-prefix=INLINEAFTER

target triple = "x86_64-unknown-linux-gnu"

; INLINEFIRST: @__profd_foo = private global{{.*}}zeroinitializer, i32 21
; INLINEFIRST: @__profd_bar = private global{{.*}}zeroinitializer, i32 23
; INLINEFIRST: @__profd_foobar = private global{{.*}}zeroinitializer, i32 99

; INLINEAFTER: @__profd_foobar = private global{{.*}}zeroinitializer, i32 99
; INLINEAFTER: @__profd_foo = private global{{.*}}zeroinitializer, i32 21
; INLINEAFTER: @__profd_bar = private global{{.*}}zeroinitializer, i32 23

; NOINLINE: @__profd_foobar = private global{{.*}}zeroinitializer, i32 99
; NOINLINE: @__profd_foo = private global{{.*}}zeroinitializer, i32 21
; NOINLINE: @__profd_bar = private global{{.*}}zeroinitializer, i32 23

declare void @llvm.instrprof.increment(ptr %0, i64 %1, i32 %2, i32 %3)
declare void @llvm.instrprof.mcdc.parameters(ptr %0, i64 %1, i32 %2)
@__profn_foobar = private constant [6 x i8] c"foobar"
@__profn_foo = private constant [3 x i8] c"foo"
@__profn_bar = private constant [3 x i8] c"bar"

define internal void @foobar() {
  call void @llvm.instrprof.increment(ptr @__profn_foobar, i64 123456, i32 32, i32 0)
  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_foobar, i64 123456, i32 99)

  ret void
}

define void @foo() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 123456, i32 32, i32 0)
  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_foo, i64 123456, i32 21)
  call void @foobar()
  ret void
}

define void @bar() {
  call void @llvm.instrprof.increment(ptr @__profn_bar, i64 123456, i32 32, i32 0)
  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_bar, i64 123456, i32 23)
  call void @foobar()
  ret void
}
