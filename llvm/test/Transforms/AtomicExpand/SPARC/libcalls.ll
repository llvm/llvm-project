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
; CHECK:  %1 = bitcast i16* %arg to i8*
; CHECK:  %2 = call i16 @__atomic_load_2(i8* %1, i32 5)
; CHECK:  ret i16 %2
define i16 @test_load_i16(i16* %arg) {
  %ret = load atomic i16, i16* %arg seq_cst, align 4
  ret i16 %ret
}

; CHECK-LABEL: @test_store_i16(
; CHECK:  %1 = bitcast i16* %arg to i8*
; CHECK:  call void @__atomic_store_2(i8* %1, i16 %val, i32 5)
; CHECK:  ret void
define void @test_store_i16(i16* %arg, i16 %val) {
  store atomic i16 %val, i16* %arg seq_cst, align 4
  ret void
}

; CHECK-LABEL: @test_exchange_i16(
; CHECK:  %1 = bitcast i16* %arg to i8*
; CHECK:  %2 = call i16 @__atomic_exchange_2(i8* %1, i16 %val, i32 5)
; CHECK:  ret i16 %2
define i16 @test_exchange_i16(i16* %arg, i16 %val) {
  %ret = atomicrmw xchg i16* %arg, i16 %val seq_cst
  ret i16 %ret
}

; CHECK-LABEL: @test_cmpxchg_i16(
; CHECK:  %1 = bitcast i16* %arg to i8*
; CHECK:  %2 = alloca i16, align 2
; CHECK:  %3 = bitcast i16* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 2, i8* %3)
; CHECK:  store i16 %old, i16* %2, align 2
; CHECK:  %4 = call zeroext i1 @__atomic_compare_exchange_2(i8* %1, i8* %3, i16 %new, i32 5, i32 0)
; CHECK:  %5 = load i16, i16* %2, align 2
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 2, i8* %3)
; CHECK:  %6 = insertvalue { i16, i1 } undef, i16 %5, 0
; CHECK:  %7 = insertvalue { i16, i1 } %6, i1 %4, 1
; CHECK:  %ret = extractvalue { i16, i1 } %7, 0
; CHECK:  ret i16 %ret
define i16 @test_cmpxchg_i16(i16* %arg, i16 %old, i16 %new) {
  %ret_succ = cmpxchg i16* %arg, i16 %old, i16 %new seq_cst monotonic
  %ret = extractvalue { i16, i1 } %ret_succ, 0
  ret i16 %ret
}

; CHECK-LABEL: @test_add_i16(
; CHECK:  %1 = bitcast i16* %arg to i8*
; CHECK:  %2 = call i16 @__atomic_fetch_add_2(i8* %1, i16 %val, i32 5)
; CHECK:  ret i16 %2
define i16 @test_add_i16(i16* %arg, i16 %val) {
  %ret = atomicrmw add i16* %arg, i16 %val seq_cst
  ret i16 %ret
}


;; Now, check the output for the unsized libcalls. i128 is used for
;; these tests because the "16" suffixed functions aren't available on
;; 32-bit i386.

; CHECK-LABEL: @test_load_i128(
; CHECK:  %1 = bitcast i128* %arg to i8*
; CHECK:  %2 = alloca i128, align 8
; CHECK:  %3 = bitcast i128* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %3)
; CHECK:  call void @__atomic_load(i32 16, i8* %1, i8* %3, i32 5)
; CHECK:  %4 = load i128, i128* %2, align 8
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %3)
; CHECK:  ret i128 %4
define i128 @test_load_i128(i128* %arg) {
  %ret = load atomic i128, i128* %arg seq_cst, align 16
  ret i128 %ret
}

; CHECK-LABEL: @test_store_i128(
; CHECK:  %1 = bitcast i128* %arg to i8*
; CHECK:  %2 = alloca i128, align 8
; CHECK:  %3 = bitcast i128* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %3)
; CHECK:  store i128 %val, i128* %2, align 8
; CHECK:  call void @__atomic_store(i32 16, i8* %1, i8* %3, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %3)
; CHECK:  ret void
define void @test_store_i128(i128* %arg, i128 %val) {
  store atomic i128 %val, i128* %arg seq_cst, align 16
  ret void
}

; CHECK-LABEL: @test_exchange_i128(
; CHECK:  %1 = bitcast i128* %arg to i8*
; CHECK:  %2 = alloca i128, align 8
; CHECK:  %3 = bitcast i128* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %3)
; CHECK:  store i128 %val, i128* %2, align 8
; CHECK:  %4 = alloca i128, align 8
; CHECK:  %5 = bitcast i128* %4 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %5)
; CHECK:  call void @__atomic_exchange(i32 16, i8* %1, i8* %3, i8* %5, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %3)
; CHECK:  %6 = load i128, i128* %4, align 8
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %5)
; CHECK:  ret i128 %6
define i128 @test_exchange_i128(i128* %arg, i128 %val) {
  %ret = atomicrmw xchg i128* %arg, i128 %val seq_cst
  ret i128 %ret
}

; CHECK-LABEL: @test_cmpxchg_i128(
; CHECK:  %1 = bitcast i128* %arg to i8*
; CHECK:  %2 = alloca i128, align 8
; CHECK:  %3 = bitcast i128* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %3)
; CHECK:  store i128 %old, i128* %2, align 8
; CHECK:  %4 = alloca i128, align 8
; CHECK:  %5 = bitcast i128* %4 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %5)
; CHECK:  store i128 %new, i128* %4, align 8
; CHECK:  %6 = call zeroext i1 @__atomic_compare_exchange(i32 16, i8* %1, i8* %3, i8* %5, i32 5, i32 0)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %5)
; CHECK:  %7 = load i128, i128* %2, align 8
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %3)
; CHECK:  %8 = insertvalue { i128, i1 } undef, i128 %7, 0
; CHECK:  %9 = insertvalue { i128, i1 } %8, i1 %6, 1
; CHECK:  %ret = extractvalue { i128, i1 } %9, 0
; CHECK:  ret i128 %ret
define i128 @test_cmpxchg_i128(i128* %arg, i128 %old, i128 %new) {
  %ret_succ = cmpxchg i128* %arg, i128 %old, i128 %new seq_cst monotonic
  %ret = extractvalue { i128, i1 } %ret_succ, 0
  ret i128 %ret
}

; This one is a verbose expansion, as there is no generic
; __atomic_fetch_add function, so it needs to expand to a cmpxchg
; loop, which then itself expands into a libcall.

; CHECK-LABEL: @test_add_i128(
; CHECK:  %1 = alloca i128, align 8
; CHECK:  %2 = alloca i128, align 8
; CHECK:  %3 = load i128, i128* %arg, align 16
; CHECK:  br label %atomicrmw.start
; CHECK:atomicrmw.start:
; CHECK:  %loaded = phi i128 [ %3, %0 ], [ %newloaded, %atomicrmw.start ]
; CHECK:  %new = add i128 %loaded, %val
; CHECK:  %4 = bitcast i128* %arg to i8*
; CHECK:  %5 = bitcast i128* %1 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %5)
; CHECK:  store i128 %loaded, i128* %1, align 8
; CHECK:  %6 = bitcast i128* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %6)
; CHECK:  store i128 %new, i128* %2, align 8
; CHECK:  %7 = call zeroext i1 @__atomic_compare_exchange(i32 16, i8* %4, i8* %5, i8* %6, i32 5, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %6)
; CHECK:  %8 = load i128, i128* %1, align 8
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %5)
; CHECK:  %9 = insertvalue { i128, i1 } undef, i128 %8, 0
; CHECK:  %10 = insertvalue { i128, i1 } %9, i1 %7, 1
; CHECK:  %success = extractvalue { i128, i1 } %10, 1
; CHECK:  %newloaded = extractvalue { i128, i1 } %10, 0
; CHECK:  br i1 %success, label %atomicrmw.end, label %atomicrmw.start
; CHECK:atomicrmw.end:
; CHECK:  ret i128 %newloaded
define i128 @test_add_i128(i128* %arg, i128 %val) {
  %ret = atomicrmw add i128* %arg, i128 %val seq_cst
  ret i128 %ret
}

;; Ensure that non-integer types get bitcast correctly on the way in and out of a libcall:

; CHECK-LABEL: @test_load_double(
; CHECK:  %1 = bitcast double* %arg to i8*
; CHECK:  %2 = call i64 @__atomic_load_8(i8* %1, i32 5)
; CHECK:  %3 = bitcast i64 %2 to double
; CHECK:  ret double %3
define double @test_load_double(double* %arg, double %val) {
  %1 = load atomic double, double* %arg seq_cst, align 16
  ret double %1
}

; CHECK-LABEL: @test_store_double(
; CHECK:  %1 = bitcast double* %arg to i8*
; CHECK:  %2 = bitcast double %val to i64
; CHECK:  call void @__atomic_store_8(i8* %1, i64 %2, i32 5)
; CHECK:  ret void
define void @test_store_double(double* %arg, double %val) {
  store atomic double %val, double* %arg seq_cst, align 16
  ret void
}

; CHECK-LABEL: @test_cmpxchg_ptr(
; CHECK:   %1 = bitcast i16** %arg to i8*
; CHECK:   %2 = alloca i16*, align 4
; CHECK:   %3 = bitcast i16** %2 to i8*
; CHECK:   call void @llvm.lifetime.start.p0i8(i64 4, i8* %3)
; CHECK:   store i16* %old, i16** %2, align 4
; CHECK:   %4 = ptrtoint i16* %new to i32
; CHECK:   %5 = call zeroext i1 @__atomic_compare_exchange_4(i8* %1, i8* %3, i32 %4, i32 5, i32 2)
; CHECK:   %6 = load i16*, i16** %2, align 4
; CHECK:   call void @llvm.lifetime.end.p0i8(i64 4, i8* %3)
; CHECK:   %7 = insertvalue { i16*, i1 } undef, i16* %6, 0
; CHECK:   %8 = insertvalue { i16*, i1 } %7, i1 %5, 1
; CHECK:   %ret = extractvalue { i16*, i1 } %8, 0
; CHECK:   ret i16* %ret
; CHECK: }
define i16* @test_cmpxchg_ptr(i16** %arg, i16* %old, i16* %new) {
  %ret_succ = cmpxchg i16** %arg, i16* %old, i16* %new seq_cst acquire
  %ret = extractvalue { i16*, i1 } %ret_succ, 0
  ret i16* %ret
}

;; ...and for a non-integer type of large size too.

; CHECK-LABEL: @test_store_fp128
; CHECK:   %1 = bitcast fp128* %arg to i8*
; CHECK:  %2 = alloca fp128, align 8
; CHECK:  %3 = bitcast fp128* %2 to i8*
; CHECK:  call void @llvm.lifetime.start.p0i8(i64 16, i8* %3)
; CHECK:  store fp128 %val, fp128* %2, align 8
; CHECK:  call void @__atomic_store(i32 16, i8* %1, i8* %3, i32 5)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 16, i8* %3)
; CHECK:  ret void
define void @test_store_fp128(fp128* %arg, fp128 %val) {
  store atomic fp128 %val, fp128* %arg seq_cst, align 16
  ret void
}

;; Unaligned loads and stores should be expanded to the generic
;; libcall, just like large loads/stores, and not a specialized one.
;; NOTE: atomicrmw and cmpxchg don't yet support an align attribute;
;; when such support is added, they should also be tested here.

; CHECK-LABEL: @test_unaligned_load_i16(
; CHECK:  __atomic_load(
define i16 @test_unaligned_load_i16(i16* %arg) {
  %ret = load atomic i16, i16* %arg seq_cst, align 1
  ret i16 %ret
}

; CHECK-LABEL: @test_unaligned_store_i16(
; CHECK: __atomic_store(
define void @test_unaligned_store_i16(i16* %arg, i16 %val) {
  store atomic i16 %val, i16* %arg seq_cst, align 1
  ret void
}
