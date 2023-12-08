; RUN: opt < %s -passes=instrprof -S | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
; CHECK: @__profc_foo = private global [1 x i8] c"\FF", section "__llvm_prf_cnts", comdat, align 1
@__profn_bar = private constant [3 x i8] c"bar"
; CHECK: @__profc_bar = private global [1 x i8] c"\FF", section "__llvm_prf_cnts", comdat, align 1

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
