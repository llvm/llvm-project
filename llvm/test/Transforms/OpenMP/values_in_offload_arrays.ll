; RUN: opt -S -passes=openmp-opt-cgscc -aa-pipeline=basic-aa -openmp-hide-memory-transfer-latency -debug-only=openmp-opt < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@.__omp_offloading_heavyComputation.region_id = weak constant i8 0
@.offload_maptypes. = private unnamed_addr constant [2 x i64] [i64 35, i64 35]

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8

; CHECK-LABEL: {{[^@]+}}Successfully got offload values:
; CHECK-NEXT: offload_baseptrs: double* %a ---   %size.addr = alloca i32, align 4 ---
; CHECK-NEXT: offload_ptrs: double* %a ---   %size.addr = alloca i32, align 4 ---
; CHECK-NEXT: offload_sizes:   %0 = shl nuw nsw i64 %conv, 3 --- i64 4 ---

;int heavyComputation(double* a, unsigned size) {
;  int random = rand() % 7;
;
;  //#pragma omp target data map(a[0:size], size)
;  void* args[2];
;  args[0] = &a;
;  args[1] = &size;
;  __tgt_target_data_begin(..., args, ...)
;
;  #pragma omp target teams
;  for (int i = 0; i < size; ++i) {
;    a[i] = ++a[i] * 3.141624;
;  }
;
;  return random;
;}
define dso_local i32 @heavyComputation(double* %a, i32 %size) {
entry:
  %size.addr = alloca i32, align 4
  %.offload_baseptrs = alloca [2 x i8*], align 8
  %.offload_ptrs = alloca [2 x i8*], align 8
  %.offload_sizes = alloca [2 x i64], align 8

  store i32 %size, i32* %size.addr, align 4
  %call = tail call i32 (...) @rand()

  %conv = zext i32 %size to i64
  %0 = shl nuw nsw i64 %conv, 3
  %1 = getelementptr inbounds [2 x i8*], [2 x i8*]* %.offload_baseptrs, i64 0, i64 0
  %2 = bitcast [2 x i8*]* %.offload_baseptrs to double**
  store double* %a, double** %2, align 8
  %3 = getelementptr inbounds [2 x i8*], [2 x i8*]* %.offload_ptrs, i64 0, i64 0
  %4 = bitcast [2 x i8*]* %.offload_ptrs to double**
  store double* %a, double** %4, align 8
  %5 = getelementptr inbounds [2 x i64], [2 x i64]* %.offload_sizes, i64 0, i64 0
  store i64 %0, i64* %5, align 8
  %6 = getelementptr inbounds [2 x i8*], [2 x i8*]* %.offload_baseptrs, i64 0, i64 1
  %7 = bitcast i8** %6 to i32**
  store i32* %size.addr, i32** %7, align 8
  %8 = getelementptr inbounds [2 x i8*], [2 x i8*]* %.offload_ptrs, i64 0, i64 1
  %9 = bitcast i8** %8 to i32**
  store i32* %size.addr, i32** %9, align 8
  %10 = getelementptr inbounds [2 x i64], [2 x i64]* %.offload_sizes, i64 0, i64 1
  store i64 4, i64* %10, align 8
  call void @__tgt_target_data_begin_mapper(%struct.ident_t* @0, i64 -1, i32 2, i8** nonnull %1, i8** nonnull %3, i64* nonnull %5, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @.offload_maptypes., i64 0, i64 0), i8** null, i8** null)
  %rem = srem i32 %call, 7
  call void @__tgt_target_data_end_mapper(%struct.ident_t* @0, i64 -1, i32 2, i8** nonnull %1, i8** nonnull %3, i64* nonnull %5, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @.offload_maptypes., i64 0, i64 0), i8** null, i8** null)
  ret i32 %rem
}

declare void @__tgt_target_data_begin_mapper(%struct.ident_t*, i64, i32, i8**, i8**, i64*, i64*, i8**, i8**)
declare void @__tgt_target_data_end_mapper(%struct.ident_t*, i64, i32, i8**, i8**, i64*, i64*, i8**, i8**)

declare dso_local i32 @rand(...)

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"openmp", i32 50}

