;; Test that we don't instrument loads to PGO counters or other
;; compiler inserted variables.
;
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

$__profc__Z3foov = comdat nodeduplicate
@__profc__Z3foov = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer

define void @_Z3foov(ptr %a) {
entry:
  ;; Load that should get instrumentation.
  %tmp1 = load i32, ptr %a, align 4
  ;; PGO counter update
  %pgocount = load i64, ptr @__profc__Z3foov, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc__Z3foov, align 8
  ;; Gcov counter update
  %gcovcount = load i64, ptr @__llvm_gcov_ctr, align 8
  %1 = add i64 %gcovcount, 1
  store i64 %1, ptr @__llvm_gcov_ctr, align 8
  ret void
}

;; We should only add memory profile instrumentation for the first load.
; CHECK: define void @_Z3foov
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = load i64, ptr @__memprof_shadow_memory_dynamic_address, align 8
; CHECK-NEXT:  %1 = ptrtoint ptr %a to i64
; CHECK-NEXT:  %2 = and i64 %1, -64
; CHECK-NEXT:  %3 = lshr i64 %2, 3
; CHECK-NEXT:  %4 = add i64 %3, %0
; CHECK-NEXT:  %5 = inttoptr i64 %4 to ptr
; CHECK-NEXT:  %6 = load i64, ptr %5, align 8
; CHECK-NEXT:  %7 = add i64 %6, 1
; CHECK-NEXT:  store i64 %7, ptr %5, align 8
; CHECK-NEXT:  %tmp1 = load i32, ptr %a, align 4
; CHECK-NEXT:  %pgocount = load i64, ptr @__profc__Z3foov
; CHECK-NEXT:  %8 = add i64 %pgocount, 1
; CHECK-NEXT:  store i64 %8, ptr @__profc__Z3foov
; CHECK-NEXT:  %gcovcount = load i64, ptr @__llvm_gcov_ctr
; CHECK-NEXT:  %9 = add i64 %gcovcount, 1
; CHECK-NEXT:  store i64 %9, ptr @__llvm_gcov_ctr
; CHECK-NEXT:  ret void
