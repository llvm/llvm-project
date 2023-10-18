; RUN: opt < %s -passes=instrprof -profile-correlate=binary -S | FileCheck %s

; CHECK: @__profd_foo = private global { i64, i64, i64, ptr, ptr, i32, [2 x i16] } { i64 {{.*}}, i64 {{.*}}, i64 ptrtoint (ptr @__profc_foo to i64)

@__profn_foo = private constant [3 x i8] c"foo"
define void @_Z3foov() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 12345678, i32 2, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
