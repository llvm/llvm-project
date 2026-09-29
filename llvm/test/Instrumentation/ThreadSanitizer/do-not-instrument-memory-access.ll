; This test checks that we are not instrumenting unwanted acesses to globals:
; - Instructions with the !nosanitize metadata (e.g. -fprofile-arcs instrumented counter accesses)
; - Instruction profiler counter instrumentation has known intended races.
;
; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

@__profc_test_gep = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
@__profc_test_bitcast = private global [2 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
@__profc_test_bitcast_foo = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8

@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer
@__llvm_gcov_ctr.1 = internal global [1 x i64] zeroinitializer
@__llvm_gcov_global_state_pred = internal global i32 0
@__llvm_gcda_foo = internal global i32 0

define i32 @test_gep() sanitize_thread {
entry:
  %pgocount = load i64, ptr @__profc_test_gep, !nosanitize !0
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_test_gep, !nosanitize !0

  %gcovcount = load i64, ptr @__llvm_gcov_ctr, !nosanitize !0
  %1 = add i64 %gcovcount, 1
  store i64 %1, ptr @__llvm_gcov_ctr, !nosanitize !0

  %gcovcount.1 = load i64, ptr @__llvm_gcov_ctr.1, !nosanitize !0
  %2 = add i64 %gcovcount.1, 1
  store i64 %2, ptr @__llvm_gcov_ctr.1, !nosanitize !0

  ret i32 1
}

define i32 @test_bitcast() sanitize_thread {
entry:
  %0 = load <2 x i64>, ptr @__profc_test_bitcast, align 8, !nosanitize !0
  %.promoted5 = load i64, ptr @__profc_test_bitcast_foo, align 8, !nosanitize !0
  %1 = add i64 %.promoted5, 10
  %2 = add <2 x i64> %0, <i64 1, i64 10>
  store <2 x i64> %2, ptr @__profc_test_bitcast, align 8, !nosanitize !0
  store i64 %1, ptr @__profc_test_bitcast_foo, align 8, !nosanitize !0
  ret i32 undef
}

define void @test_load() sanitize_thread {
entry:
  %0 = load i32, ptr @__llvm_gcov_global_state_pred, !nosanitize !0
  store i32 1, ptr @__llvm_gcov_global_state_pred, !nosanitize !0

  %1 = load i32, ptr @__llvm_gcda_foo, !nosanitize !0
  store i32 1, ptr @__llvm_gcda_foo, !nosanitize !0

  ret void
}

!0 = !{}

; CHECK-NOT: {{call void @__tsan_write}}
; CHECK: __tsan_init
