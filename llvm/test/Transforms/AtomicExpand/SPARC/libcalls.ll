; RUN: opt -S %s -atomic-expand | FileCheck %s

;;; NOTE: this test is actually target-independent -- any target which
;;; doesn't support inline atomics can be used. (E.g. X86 i386 would
;;; work, if LLVM is properly taught about what it's missing vs i586.)

;target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
;target triple = "i386-unknown-unknown"
target datalayout = "e-m:e-p:32:32-i64:64-f128:64-n32-S64"
target triple = "sparc-unknown-unknown"

;; First, check the sized calls. Except for cmpxchg, these are fairly
;; straightforward.

; CHECK-LABEL: @test_load_i16(
; CHECK:  %1 = call i16 @__atomic_load_2(ptr %arg, i32 5)
; CHECK:  ret i16 %1
define i16 @test_load_i16(ptr %arg) {
  %ret = load atomic i16, ptr %arg seq_cst, align 4
  ret i16 %ret
}

; CHECK-LABEL: @test_store_i16(
; CHECK:  call void @__atomic_store_2(ptr %arg, i16 %val, i32 5)
; CHECK:  ret void
define void @test_store_i16(ptr %arg, i16 %val) {
  store atomic i16 %val, ptr %arg seq_cst, align 4
  ret void
}

; CHECK-LABEL: @test_exchange_i16(
; CHECK:  %1 = call i16 @__atomic_exchange_2(ptr %arg, i16 %val, i32 5)
; CHECK:  ret i16 %1
define i16 @test_exchange_i16(ptr %arg, i16 %val) {
  %ret = atomicrmw xchg ptr %arg, i16 %val seq_cst
  ret i16 %ret
}

; CHECK-LABEL: @test_cmpxchg_i16(
; CHECK:  %1 = alloca i16, align 2
; CHECK:  call void @llvm.lifetime.start.p0(i64 2, ptr %1)
; CHECK:  store i16 %old, ptr %1, align 2
; CHECK:  %2 = call zeroext i1 @__atomic_compare_exchange_2(ptr %arg, ptr %1, i16 %new, i32 5, i32 0)
; CHECK:  %3 = load i16, ptr %1, align 2
; CHECK:  call void @llvm.lifetime.end.p0(i64 2, ptr %1)
; CHECK:  %4 = insertvalue { i16, i1 } poison, i16 %3, 0
; CHECK:  %5 = insertvalue { i16, i1 } %4, i1 %2, 1
; CHECK:  %ret = extractvalue { i16, i1 } %5, 0
; CHECK:  ret i16 %ret
define i16 @test_cmpxchg_i16(ptr %arg, i16 %old, i16 %new) {
  %ret_succ = cmpxchg ptr %arg, i16 %old, i16 %new seq_cst monotonic
  %ret = extractvalue { i16, i1 } %ret_succ, 0
  ret i16 %ret
}

; CHECK-LABEL: @test_add_i16(
; CHECK:  %1 = call i16 @__atomic_fetch_add_2(ptr %arg, i16 %val, i32 5)
; CHECK:  ret i16 %1
define i16 @test_add_i16(ptr %arg, i16 %val) {
  %ret = atomicrmw add ptr %arg, i16 %val seq_cst
  ret i16 %ret
}


;; Now, check the output for the unsized libcalls. i128 is used for
;; these tests because the "16" suffixed functions aren't available on
;; 32-bit i386.

; CHECK-LABEL: @test_load_i128(
; CHECK:  %1 = alloca i128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %1)
; CHECK:  call void @__atomic_load(i32 16, ptr %arg, ptr %1, i32 5)
; CHECK:  %2 = load i128, ptr %1, align 8
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %1)
; CHECK:  ret i128 %2
define i128 @test_load_i128(ptr %arg) {
  %ret = load atomic i128, ptr %arg seq_cst, align 16
  ret i128 %ret
}

; CHECK-LABEL: @test_store_i128(
; CHECK:  %1 = alloca i128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %1)
; CHECK:  store i128 %val, ptr %1, align 8
; CHECK:  call void @__atomic_store(i32 16, ptr %arg, ptr %1, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %1)
; CHECK:  ret void
define void @test_store_i128(ptr %arg, i128 %val) {
  store atomic i128 %val, ptr %arg seq_cst, align 16
  ret void
}

; CHECK-LABEL: @test_exchange_i128(
; CHECK:  %1 = alloca i128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %1)
; CHECK:  store i128 %val, ptr %1, align 8
; CHECK:  %2 = alloca i128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %2)
; CHECK:  call void @__atomic_exchange(i32 16, ptr %arg, ptr %1, ptr %2, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %1)
; CHECK:  %3 = load i128, ptr %2, align 8
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %2)
; CHECK:  ret i128 %3
define i128 @test_exchange_i128(ptr %arg, i128 %val) {
  %ret = atomicrmw xchg ptr %arg, i128 %val seq_cst
  ret i128 %ret
}

; CHECK-LABEL: @test_cmpxchg_i128(
; CHECK:  %1 = alloca i128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %1)
; CHECK:  store i128 %old, ptr %1, align 8
; CHECK:  %2 = alloca i128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %2)
; CHECK:  store i128 %new, ptr %2, align 8
; CHECK:  %3 = call zeroext i1 @__atomic_compare_exchange(i32 16, ptr %arg, ptr %1, ptr %2, i32 5, i32 0)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %2)
; CHECK:  %4 = load i128, ptr %1, align 8
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %1)
; CHECK:  %5 = insertvalue { i128, i1 } poison, i128 %4, 0
; CHECK:  %6 = insertvalue { i128, i1 } %5, i1 %3, 1
; CHECK:  %ret = extractvalue { i128, i1 } %6, 0
; CHECK:  ret i128 %ret
define i128 @test_cmpxchg_i128(ptr %arg, i128 %old, i128 %new) {
  %ret_succ = cmpxchg ptr %arg, i128 %old, i128 %new seq_cst monotonic
  %ret = extractvalue { i128, i1 } %ret_succ, 0
  ret i128 %ret
}

; This one is a verbose expansion, as there is no generic
; __atomic_fetch_add function, so it needs to expand to a cmpxchg
; loop, which then itself expands into a libcall.

; CHECK-LABEL: @test_add_i128(
; CHECK:  %1 = alloca i128, align 8
; CHECK:  %2 = alloca i128, align 8
; CHECK:  %3 = load i128, ptr %arg, align 16
; CHECK:  br label %atomicrmw.start
; CHECK:atomicrmw.start:
; CHECK:  %loaded = phi i128 [ %3, %0 ], [ %newloaded, %atomicrmw.start ]
; CHECK:  %new = add i128 %loaded, %val
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %1)
; CHECK:  store i128 %loaded, ptr %1, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %2)
; CHECK:  store i128 %new, ptr %2, align 8
; CHECK:  %4 = call zeroext i1 @__atomic_compare_exchange(i32 16, ptr %arg, ptr %1, ptr %2, i32 5, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %2)
; CHECK:  %5 = load i128, ptr %1, align 8
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %1)
; CHECK:  %6 = insertvalue { i128, i1 } poison, i128 %5, 0
; CHECK:  %7 = insertvalue { i128, i1 } %6, i1 %4, 1
; CHECK:  %success = extractvalue { i128, i1 } %7, 1
; CHECK:  %newloaded = extractvalue { i128, i1 } %7, 0
; CHECK:  br i1 %success, label %atomicrmw.end, label %atomicrmw.start
; CHECK:atomicrmw.end:
; CHECK:  ret i128 %newloaded
define i128 @test_add_i128(ptr %arg, i128 %val) {
  %ret = atomicrmw add ptr %arg, i128 %val seq_cst
  ret i128 %ret
}

;; Ensure that non-integer types get bitcast correctly on the way in and out of a libcall:

; CHECK-LABEL: @test_load_double(
; CHECK:  %1 = call i64 @__atomic_load_8(ptr %arg, i32 5)
; CHECK:  %2 = bitcast i64 %1 to double
; CHECK:  ret double %2
define double @test_load_double(ptr %arg, double %val) {
  %1 = load atomic double, ptr %arg seq_cst, align 16
  ret double %1
}

; CHECK-LABEL: @test_store_double(
; CHECK:  %1 = bitcast double %val to i64
; CHECK:  call void @__atomic_store_8(ptr %arg, i64 %1, i32 5)
; CHECK:  ret void
define void @test_store_double(ptr %arg, double %val) {
  store atomic double %val, ptr %arg seq_cst, align 16
  ret void
}

; CHECK-LABEL: @test_cmpxchg_ptr(
; CHECK:   %1 = alloca ptr, align 4
; CHECK:   call void @llvm.lifetime.start.p0(i64 4, ptr %1)
; CHECK:   store ptr %old, ptr %1, align 4
; CHECK:   %2 = ptrtoint ptr %new to i32
; CHECK:   %3 = call zeroext i1 @__atomic_compare_exchange_4(ptr %arg, ptr %1, i32 %2, i32 5, i32 2)
; CHECK:   %4 = load ptr, ptr %1, align 4
; CHECK:   call void @llvm.lifetime.end.p0(i64 4, ptr %1)
; CHECK:   %5 = insertvalue { ptr, i1 } poison, ptr %4, 0
; CHECK:   %6 = insertvalue { ptr, i1 } %5, i1 %3, 1
; CHECK:   %ret = extractvalue { ptr, i1 } %6, 0
; CHECK:   ret ptr %ret
; CHECK: }
define ptr @test_cmpxchg_ptr(ptr %arg, ptr %old, ptr %new) {
  %ret_succ = cmpxchg ptr %arg, ptr %old, ptr %new seq_cst acquire
  %ret = extractvalue { ptr, i1 } %ret_succ, 0
  ret ptr %ret
}

;; ...and for a non-integer type of large size too.

; CHECK-LABEL: @test_store_fp128
; CHECK:  %1 = alloca fp128, align 8
; CHECK:  call void @llvm.lifetime.start.p0(i64 16, ptr %1)
; CHECK:  store fp128 %val, ptr %1, align 8
; CHECK:  call void @__atomic_store(i32 16, ptr %arg, ptr %1, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr %1)
; CHECK:  ret void
define void @test_store_fp128(ptr %arg, fp128 %val) {
  store atomic fp128 %val, ptr %arg seq_cst, align 16
  ret void
}

;; Unaligned loads and stores should be expanded to the generic
;; libcall, just like large loads/stores, and not a specialized one.
;; NOTE: atomicrmw and cmpxchg don't yet support an align attribute;
;; when such support is added, they should also be tested here.

; CHECK-LABEL: @test_unaligned_load_i16(
; CHECK:  __atomic_load(
define i16 @test_unaligned_load_i16(ptr %arg) {
  %ret = load atomic i16, ptr %arg seq_cst, align 1
  ret i16 %ret
}

; CHECK-LABEL: @test_unaligned_store_i16(
; CHECK: __atomic_store(
define void @test_unaligned_store_i16(ptr %arg, i16 %val) {
  store atomic i16 %val, ptr %arg seq_cst, align 1
  ret void
}
