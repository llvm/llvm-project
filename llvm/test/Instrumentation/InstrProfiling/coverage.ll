; RUN: opt < %s -passes=instrprof -S | FileCheck %s
; RUN: opt < %s -passes=instrprof -profile-correlate=binary -S | FileCheck %s --check-prefix=BINARY

target triple = "aarch64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
; CHECK: @__profc_foo = private global [1 x i8] c"\FF", section "__llvm_prf_cnts", comdat, align 1
; CHECK: @__profd_foo = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 {{.*}}, i64 {{.*}}, i64 sub (i64 ptrtoint (ptr @__profc_foo to i64)
; BINARY: @__profd_foo = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 {{.*}}, i64 {{.*}}, i64 ptrtoint (ptr @__profc_foo to i64),
@__profn_bar = private constant [3 x i8] c"bar"
; CHECK: @__profc_bar = private global [1 x i8] c"\FF", section "__llvm_prf_cnts", comdat, align 1
; CHECK: @__profd_bar = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 {{.*}}, i64 {{.*}}, i64 sub (i64 ptrtoint (ptr @__profc_bar to i64)
; BINARY: @__profd_bar = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 {{.*}}, i64 {{.*}}, i64 ptrtoint (ptr @__profc_bar to i64),

; CHECK: @__llvm_prf_nm = {{.*}} section "__llvm_prf_names"
; BINARY: @__llvm_prf_nm ={{.*}} section "__llvm_covnames"

define void @_Z3foov() {
  call void @llvm.instrprof.cover(ptr @__profn_foo, i64 12345678, i32 1, i32 0)
  ; CHECK: store i8 0, ptr @__profc_foo, align 1
  ret void
}

%class.A = type { ptr }
define dso_local void @_Z3barv(ptr nocapture nonnull align 8 %0) unnamed_addr #0 align 2 {
  call void @llvm.instrprof.cover(ptr @__profn_bar, i64 87654321, i32 1, i32 0)
  ; CHECK: store i8 0, ptr @__profc_bar, align 1
  ret void
}

declare void @llvm.instrprof.cover(ptr, i64, i32, i32)
