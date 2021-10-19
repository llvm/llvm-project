; ASan shouldn't instrument code added by UBSan.

; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 (...)** }
declare void @__ubsan_handle_dynamic_type_cache_miss(i8*, i64, i64) uwtable
@__ubsan_vptr_type_cache = external global [128 x i64]
@.src = private unnamed_addr constant [19 x i8] c"tmp/ubsan/vptr.cpp\00", align 1
@0 = private unnamed_addr constant { i16, i16, [4 x i8] } { i16 -1, i16 0, [4 x i8] c"'A'\00" }
@_ZTI1A = external constant i8*
@1 = private unnamed_addr global { { [19 x i8]*, i32, i32 }, { i16, i16, [4 x i8] }*, i8*, i8 } { { [19 x i8]*, i32, i32 } { [19 x i8]* @.src, i32 2, i32 18 }, { i16, i16, [4 x i8] }* @0, i8* bitcast (i8** @_ZTI1A to i8*), i8 4 }

define void @_Z3BarP1A(%struct.A* %a) uwtable sanitize_address {
; CHECK-LABEL: define void @_Z3BarP1A
entry:
  %0 = bitcast %struct.A* %a to void (%struct.A*)***
  %vtable = load void (%struct.A*)**, void (%struct.A*)*** %0, align 8
; CHECK: __asan_report_load8
  %1 = load void (%struct.A*)*, void (%struct.A*)** %vtable, align 8
; CHECK: __asan_report_load8
  %2 = ptrtoint void (%struct.A*)** %vtable to i64
  %3 = xor i64 %2, -303164226014115343, !nosanitize !0
  %4 = mul i64 %3, -7070675565921424023, !nosanitize !0
  %5 = lshr i64 %4, 47, !nosanitize !0
  %6 = xor i64 %4, %2, !nosanitize !0
  %7 = xor i64 %6, %5, !nosanitize !0
  %8 = mul i64 %7, -7070675565921424023, !nosanitize !0
  %9 = lshr i64 %8, 47, !nosanitize !0
  %10 = xor i64 %9, %8, !nosanitize !0
  %11 = mul i64 %10, -7070675565921424023, !nosanitize !0
  %12 = and i64 %11, 127, !nosanitize !0
  %13 = getelementptr inbounds [128 x i64], [128 x i64]* @__ubsan_vptr_type_cache, i64 0, i64 %12, !nosanitize !0
; CHECK-NOT: __asan_report_load8
  %14 = load i64, i64* %13, align 8, !nosanitize !0
  %15 = icmp eq i64 %14, %11, !nosanitize !0
  br i1 %15, label %cont, label %handler.dynamic_type_cache_miss, !nosanitize !0

handler.dynamic_type_cache_miss:                  ; preds = %entry
  %16 = ptrtoint %struct.A* %a to i64, !nosanitize !0
  tail call void @__ubsan_handle_dynamic_type_cache_miss(i8* bitcast ({ { [19 x i8]*, i32, i32 }, { i16, i16, [4 x i8] }*, i8*, i8 }* @1 to i8*), i64 %16, i64 %11) #2, !nosanitize !0
  br label %cont, !nosanitize !0

cont:                                             ; preds = %handler.dynamic_type_cache_miss, %entry
  tail call void %1(%struct.A* %a)
; CHECK: ret void
  ret void
}

!0 = !{}
