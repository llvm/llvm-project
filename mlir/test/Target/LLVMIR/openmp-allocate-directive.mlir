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
