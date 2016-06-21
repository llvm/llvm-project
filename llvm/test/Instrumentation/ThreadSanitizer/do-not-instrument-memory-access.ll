; This test checks that we are not instrumenting unwanted acesses to globals:
; - Instruction profiler counter instrumentation has known intended races.
; - The gcov counters array has a known intended race.
;
; RUN: opt < %s -tsan -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

@__profc_test_gep = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
@__profc_test_bitcast = private global [2 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
@__profc_test_bitcast_foo = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8

@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer

define i32 @test_gep() sanitize_thread {
entry:
  %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test_gep, i64 0, i64 0)
  %0 = add i64 %pgocount, 1
  store i64 %0, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test_gep, i64 0, i64 0)

  %gcovcount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0)
  %1 = add i64 %gcovcount, 1
  store i64 %1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0)

  ret i32 1
}

define i32 @test_bitcast() sanitize_thread {
entry:
  %0 = load <2 x i64>, <2 x i64>* bitcast ([2 x i64]* @__profc_test_bitcast to <2 x i64>*), align 8
  %.promoted5 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test_bitcast_foo, i64 0, i64 0), align 8
  %1 = add i64 %.promoted5, 10
  %2 = add <2 x i64> %0, <i64 1, i64 10>
  store <2 x i64> %2, <2 x i64>* bitcast ([2 x i64]* @__profc_test_bitcast to <2 x i64>*), align 8
  store i64 %1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test_bitcast_foo, i64 0, i64 0), align 8
  ret i32 undef
}

; CHECK-NOT: {{call void @__tsan_write}}
; CHECK: __tsan_init
