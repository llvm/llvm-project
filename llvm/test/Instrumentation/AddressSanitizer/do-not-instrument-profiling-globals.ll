; This test checks that we don't instrument globals created by profiling passes.
; RUN: opt < %s -asan -asan-module -S | FileCheck %s

@__profc_test = private global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer

; CHECK-DAG: @asan.module_ctor
; CHECK-NOT: @___asan_gen{{.*}}__llvm_gcov_ctr
; CHECK-NOT: @___asan_gen{{.*}}__profc_test
