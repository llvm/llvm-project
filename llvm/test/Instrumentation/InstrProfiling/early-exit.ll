; RUN: opt < %s -passes=instrprof -S | FileCheck %s

target triple = "x86_64-apple-macosx10.10.0"

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

declare void @llvm.instrprof.increment.step(ptr, i64, i32, i32, i64)

; CHECK: @__profc_foo = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
@__profn_foo = private constant [3 x i8] c"foo"

define void @foo() {
  call void @llvm.instrprof.increment.step(ptr @__profn_foo, i64 0, i32 1, i32 0, i64 0)
  ret void
}
