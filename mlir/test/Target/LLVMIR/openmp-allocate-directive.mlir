// Tests for translation of omp.allocate_dir / omp.allocate_free pairs to
// LLVM IR, covering all combinations of align and allocator clauses.
// The frontend is responsible for placing omp.allocate_free at the correct
// Fortran scope exit; here each function pairs the ops manually.

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: define void @test_allocate_default
// CHECK-SAME: (ptr %[[ARG0:.*]]) {
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_alloc(i32 %[[TID]], i64 8, ptr null)
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr null)
// CHECK:   ret void
// CHECK: }
// CHECK: declare noalias ptr @__kmpc_alloc(i32, i64, ptr)
// CHECK: declare void @__kmpc_free(i32, ptr, ptr)
llvm.func @test_allocate_default(%arg0: !llvm.ptr) {
  omp.allocate_dir (%arg0 : !llvm.ptr)
  omp.allocate_free (%arg0 : !llvm.ptr)
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_allocate_align_only
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID]], i64 16, i64 16, ptr null)
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr null)
// CHECK:   ret void
// CHECK: declare noalias ptr @__kmpc_aligned_alloc(i32, i64, i64, ptr)
llvm.func @test_allocate_align_only(%arg0: !llvm.ptr) {
  omp.allocate_dir (%arg0 : !llvm.ptr) align(16)
  omp.allocate_free (%arg0 : !llvm.ptr)
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_allocate_allocator_only
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_alloc(i32 %[[TID]], i64 8, ptr inttoptr (i32 1 to ptr))
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr inttoptr (i32 1 to ptr))
// CHECK:   ret void
llvm.func @test_allocate_allocator_only(%arg0: !llvm.ptr) {
  %alloc1 = llvm.mlir.constant(1 : i32) : i32
  omp.allocate_dir (%arg0 : !llvm.ptr) allocator(%alloc1 : i32)
  omp.allocate_free (%arg0 : !llvm.ptr) allocator(%alloc1 : i32)
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_allocate_align_and_allocator
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID]], i64 64, i64 64, ptr inttoptr (i32 6 to ptr))
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr inttoptr (i32 6 to ptr))
// CHECK:   ret void
llvm.func @test_allocate_align_and_allocator(%arg0: !llvm.ptr) {
  %alloc6 = llvm.mlir.constant(6 : i32) : i32
  omp.allocate_dir (%arg0 : !llvm.ptr) align(64) allocator(%alloc6 : i32)
  omp.allocate_free (%arg0 : !llvm.ptr) allocator(%alloc6 : i32)
  llvm.return
}

// -----

// Verifies that multiple variables each get their own __kmpc_aligned_alloc
// call, and that __kmpc_free calls are emitted in reverse allocation order.
//
// CHECK-LABEL: define void @test_allocate_multiple_vars
// CHECK:   %[[TID0:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC0:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID0]], i64 32, i64 32, ptr inttoptr (i32 3 to ptr))
// CHECK:   %[[TID1:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC1:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID1]], i64 32, i64 32, ptr inttoptr (i32 3 to ptr))
// CHECK:   %[[TID2:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC2:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID2]], i64 32, i64 32, ptr inttoptr (i32 3 to ptr))
// Free order is reversed relative to allocation order.
// CHECK:   call void @__kmpc_free({{.*}}, ptr %[[ALLOC2]], ptr inttoptr (i32 3 to ptr))
// CHECK:   call void @__kmpc_free({{.*}}, ptr %[[ALLOC1]], ptr inttoptr (i32 3 to ptr))
// CHECK:   call void @__kmpc_free({{.*}}, ptr %[[ALLOC0]], ptr inttoptr (i32 3 to ptr))
// CHECK:   ret void
llvm.func @test_allocate_multiple_vars(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %alloc3 = llvm.mlir.constant(3 : i32) : i32
  omp.allocate_dir (%arg0, %arg1, %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) align(32) allocator(%alloc3 : i32)
  omp.allocate_free (%arg0, %arg1, %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) allocator(%alloc3 : i32)
  llvm.return
}

// -----

// Verifies that array size is correctly calculated from the global's element
// type: [10 x i32] = 40 bytes, rounded up to alignment 64 => 64 bytes.
//
// CHECK-LABEL: define void @test_allocate_array_global
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID]], i64 64, i64 64, ptr inttoptr (i32 6 to ptr))
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr inttoptr (i32 6 to ptr))
// CHECK:   ret void
llvm.mlir.global internal @arr_global() : !llvm.array<10 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<10 x i32>
  llvm.return %0 : !llvm.array<10 x i32>
}

llvm.func @test_allocate_array_global() {
  %z = llvm.mlir.addressof @arr_global : !llvm.ptr
  %alloc6 = llvm.mlir.constant(6 : i32) : i32
  omp.allocate_dir (%z : !llvm.ptr) align(64) allocator(%alloc6 : i32)
  omp.allocate_free (%z : !llvm.ptr) allocator(%alloc6 : i32)
  llvm.return
}

// -----

// Verifies that array size is correctly calculated from a stack alloca:
// [10 x i32] = 40 bytes, rounded up to alignment 64 => 64 bytes.
//
// CHECK-LABEL: define void @test_allocate_array_stack
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %[[TID]], i64 64, i64 64, ptr null)
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr null)
// CHECK:   ret void
llvm.func @test_allocate_array_stack() {
  %one = llvm.mlir.constant(1 : i64) : i64
  %arr = llvm.alloca %one x !llvm.array<10 x i32> : (i64) -> !llvm.ptr
  omp.allocate_dir (%arr : !llvm.ptr) align(64)
  omp.allocate_free (%arr : !llvm.ptr)
  llvm.return
}

// -----

// Verifies that loads and stores after omp.allocate_dir use the OMP-allocated
// pointer rather than the original storage.
//
// CHECK-LABEL: define void @test_allocate_use
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_alloc(i32 %[[TID]], i64 8, ptr null)
// CHECK:   store i32 42, ptr %[[ALLOC]]
// CHECK:   %[[VAL:.*]] = load i32, ptr %[[ALLOC]]
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr null)
// CHECK:   ret void
llvm.func @test_allocate_use(%arg0: !llvm.ptr) {
  omp.allocate_dir (%arg0 : !llvm.ptr)
  %c42 = llvm.mlir.constant(42 : i32) : i32
  llvm.store %c42, %arg0 : i32, !llvm.ptr
  %v = llvm.load %arg0 : !llvm.ptr -> i32
  omp.allocate_free (%arg0 : !llvm.ptr)
  llvm.return
}

// -----

// Verifies remapping when a global has multiple GEP users (COMMON block shape).
//
// CHECK-LABEL: define void @test_allocate_global_gep_users
// CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   %[[ALLOC:.*]] = call ptr @__kmpc_alloc(i32 %[[TID]], i64 8, ptr null)
// CHECK:   %[[GEP4:.*]] = getelementptr i8, ptr %[[ALLOC]], i64 4
// CHECK:   store i32 1, ptr %[[ALLOC]], align 4
// CHECK:   store i32 2, ptr %[[GEP4]], align 4
// CHECK:   %[[TID_FREE:.*]] = call i32 @__kmpc_global_thread_num(
// CHECK:   call void @__kmpc_free(i32 %[[TID_FREE]], ptr %[[ALLOC]], ptr null)
// CHECK:   ret void
llvm.mlir.global internal @common_like() : !llvm.array<2 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<2 x i32>
  llvm.return %0 : !llvm.array<2 x i32>
}

llvm.func @test_allocate_global_gep_users() {
  %base = llvm.mlir.addressof @common_like : !llvm.ptr
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %m0 = llvm.getelementptr %base[%c0, %c0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i32>
  %m1 = llvm.getelementptr %base[%c0, %c1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i32>
  omp.allocate_dir (%base : !llvm.ptr)
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  llvm.store %one, %m0 : i32, !llvm.ptr
  llvm.store %two, %m1 : i32, !llvm.ptr
  omp.allocate_free (%base : !llvm.ptr)
  llvm.return
}
