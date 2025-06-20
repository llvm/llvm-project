; ASan shouldn't instrument code added by UBSan.

; RUN: opt < %s -passes=asan -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-detect-invalid-pointer-cmp -S | FileCheck %s --check-prefixes=NOCMP

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }
declare void @__ubsan_handle_dynamic_type_cache_miss(ptr, i64, i64) uwtable
declare void @__ubsan_handle_pointer_overflow(ptr, i64, i64) uwtable
@__ubsan_vptr_type_cache = external global [128 x i64]
@.src = private unnamed_addr constant [19 x i8] c"tmp/ubsan/vptr.cpp\00", align 1
@0 = private unnamed_addr constant { i16, i16, [4 x i8] } { i16 -1, i16 0, [4 x i8] c"'A'\00" }
@_ZTI1A = external constant ptr
@1 = private unnamed_addr global { { ptr, i32, i32 }, ptr, ptr, i8 } { { ptr, i32, i32 } { ptr @.src, i32 2, i32 18 }, ptr @0, ptr @_ZTI1A, i8 4 }
@2 = private unnamed_addr global { { ptr, i32, i32 } } { { ptr, i32, i32 } { ptr @.src, i32 24, i32 25 } }

define void @_Z3BarP1A(ptr %a) uwtable sanitize_address {
; CHECK-LABEL: define void @_Z3BarP1A
entry:
  %vtable = load ptr, ptr %a, align 8
; CHECK: __asan_report_load8
  %0 = load ptr, ptr %vtable, align 8
; CHECK: __asan_report_load8
  %1 = ptrtoint ptr %vtable to i64
  %2 = xor i64 %1, -303164226014115343, !nosanitize !0
  %3 = mul i64 %2, -7070675565921424023, !nosanitize !0
  %4 = lshr i64 %3, 47, !nosanitize !0
  %5 = xor i64 %3, %1, !nosanitize !0
  %6 = xor i64 %5, %4, !nosanitize !0
  %7 = mul i64 %6, -7070675565921424023, !nosanitize !0
  %8 = lshr i64 %7, 47, !nosanitize !0
  %9 = xor i64 %8, %7, !nosanitize !0
  %10 = mul i64 %9, -7070675565921424023, !nosanitize !0
  %11 = and i64 %10, 127, !nosanitize !0
  %12 = getelementptr inbounds [128 x i64], ptr @__ubsan_vptr_type_cache, i64 0, i64 %11, !nosanitize !0
; CHECK-NOT: __asan_report_load8
  %13 = load i64, ptr %12, align 8, !nosanitize !0
  %14 = icmp eq i64 %13, %10, !nosanitize !0
  br i1 %14, label %cont, label %handler.dynamic_type_cache_miss, !nosanitize !0

handler.dynamic_type_cache_miss:                  ; preds = %entry
  %15 = ptrtoint ptr %a to i64, !nosanitize !0
  tail call void @__ubsan_handle_dynamic_type_cache_miss(ptr @1, i64 %15, i64 %10) #2, !nosanitize !0
  br label %cont, !nosanitize !0

cont:                                             ; preds = %handler.dynamic_type_cache_miss, %entry
  tail call void %0(ptr %a)
; CHECK: ret void
  ret void
}

define void @_Z3foov() uwtable sanitize_address {
; NOCMP-LABEL: define void @_Z3foov
entry:
  %bar = alloca [10 x i8], align 1
  %arrayidx = getelementptr inbounds [10 x i8], ptr %bar, i64 0, i64 4
  %0 = ptrtoint ptr %bar to i64, !nosanitize !0
; NOCMP-NOT: call void @__sanitizer_ptr_cmp
  %1 = icmp ult ptr %bar, inttoptr (i64 -4 to ptr), !nosanitize !0
  br i1 %1, label %cont, label %handler.pointer_overflow, !nosanitize !0

handler.pointer_overflow:                         ; preds = %entry
  %2 = add i64 %0, 4, !nosanitize !0
  call void @__ubsan_handle_pointer_overflow(ptr @2, i64 %0, i64 %2), !nosanitize !0
  br label %cont, !nosanitize !0

cont:                                             ; preds = %handler.pointer_overflow, %entry
  store i8 0, ptr %arrayidx, align 1
; NOCMP: ret void
  ret void
}

!0 = !{}
