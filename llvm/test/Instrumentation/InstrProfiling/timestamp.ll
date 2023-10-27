; RUN: opt < %s -passes=instrprof -S | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
; CHECK: @__profc_foo = private global [2 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8

define void @_Z3foov() {
  call void @llvm.instrprof.timestamp(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12345678, i32 2, i32 0)
  ; CHECK: call void @__llvm_profile_set_timestamp(ptr @__profc_foo)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12345678, i32 2, i32 1)
  ret void
}

declare void @llvm.instrprof.timestamp(i8*, i64, i32, i32)
declare void @llvm.instrprof.increment(i8*, i64, i32, i32)
