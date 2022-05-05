;; Test that we don't instrument loads to PGO counters or other
;; compiler inserted variables.
;
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

$__profc__Z3foov = comdat nodeduplicate
@__profc__Z3foov = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer

define void @_Z3foov(i32* %a) {
entry:
  ;; Load that should get instrumentation.
  %tmp1 = load i32, i32* %a, align 4
  ;; PGO counter update
  %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__Z3foov, i64 0, i64 0), align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__Z3foov, i64 0, i64 0), align 8
  ;; Gcov counter update
  %gcovcount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0), align 8
  %1 = add i64 %gcovcount, 1
  store i64 %1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0), align 8
  ret void
}

;; We should only add memory profile instrumentation for the first load.
; CHECK: define void @_Z3foov
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = load i64, i64* @__memprof_shadow_memory_dynamic_address, align 8
; CHECK-NEXT:  %1 = ptrtoint i32* %a to i64
; CHECK-NEXT:  %2 = and i64 %1, -64
; CHECK-NEXT:  %3 = lshr i64 %2, 3
; CHECK-NEXT:  %4 = add i64 %3, %0
; CHECK-NEXT:  %5 = inttoptr i64 %4 to i64*
; CHECK-NEXT:  %6 = load i64, i64* %5, align 8
; CHECK-NEXT:  %7 = add i64 %6, 1
; CHECK-NEXT:  store i64 %7, i64* %5, align 8
; CHECK-NEXT:  %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:  %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__Z3foov, i64 0, i64 0)
; CHECK-NEXT:  %8 = add i64 %pgocount, 1
; CHECK-NEXT:  store i64 %8, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__Z3foov, i64 0, i64 0)
; CHECK-NEXT:  %gcovcount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0)
; CHECK-NEXT:  %9 = add i64 %gcovcount, 1
; CHECK-NEXT:  store i64 %9, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0)
; CHECK-NEXT:  ret void
