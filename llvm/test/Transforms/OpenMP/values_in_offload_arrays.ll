; RUN: opt -S -passes=openmp-opt-cgscc -aa-pipeline=basic-aa -openmp-hide-memory-transfer-latency -debug-only=openmp-opt < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@.__omp_offloading_heavyComputation.region_id = weak constant i8 0
@.offload_maptypes. = private unnamed_addr constant [2 x i64] [i64 35, i64 35]

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @.str }, align 8

; CHECK-LABEL: {{[^@]+}}Successfully got offload values:
; CHECK-NEXT: offload_baseptrs: ptr %a ---   %size.addr = alloca i32, align 4 ---
; CHECK-NEXT: offload_ptrs: ptr %a ---   %size.addr = alloca i32, align 4 ---
; CHECK-NEXT: offload_sizes:   %0 = shl nuw nsw i64 %conv, 3 --- i64 4 ---

;int heavyComputation(ptr a, unsigned size) {
;  int random = rand() % 7;
;
;  //#pragma omp target data map(a[0:size], size)
;  ptr args[2];
;  args[0] = &a;
;  args[1] = &size;
;  __tgt_target_data_begin(..., args, ...)
;
;  #pragma omp target teams
;  for (int i = 0; i < size; ++i) {
;    a[i] = ++aptr 3.141624;
;  }
;
;  return random;
;}
define dso_local i32 @heavyComputation(ptr %a, i32 %size) {
entry:
  %size.addr = alloca i32, align 4
  %.offload_baseptrs = alloca [2 x ptr], align 8
  %.offload_ptrs = alloca [2 x ptr], align 8
  %.offload_sizes = alloca [2 x i64], align 8

  store i32 %size, ptr %size.addr, align 4
  %call = tail call i32 (...) @rand()

  %conv = zext i32 %size to i64
  %0 = shl nuw nsw i64 %conv, 3
  store ptr %a, ptr %.offload_baseptrs, align 8
  store ptr %a, ptr %.offload_ptrs, align 8
  store i64 %0, ptr %.offload_sizes, align 8
  %1 = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i64 0, i64 1
  store ptr %size.addr, ptr %1, align 8
  %2 = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i64 0, i64 1
  store ptr %size.addr, ptr %2, align 8
  %3 = getelementptr inbounds [2 x i64], ptr %.offload_sizes, i64 0, i64 1
  store i64 4, ptr %3, align 8
  call void @__tgt_target_data_begin_mapper(ptr @0, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr @.offload_maptypes., ptr null, ptr null)
  %rem = srem i32 %call, 7
  call void @__tgt_target_data_end_mapper(ptr @0, i64 -1, i32 2, ptr nonnull %.offload_baseptrs, ptr nonnull %.offload_ptrs, ptr nonnull %.offload_sizes, ptr @.offload_maptypes., ptr null, ptr null)
  ret i32 %rem
}

declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare dso_local i32 @rand(...)

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"openmp", i32 50}

