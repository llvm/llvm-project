// RUN: mlir-opt %s -convert-openmp-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s

// This file contains MLIR test cases for omp.target_allocmem and omp.target_freemem

// CHECK-LABEL: test_alloc_free_i64
// CHECK: %[[ALLOC:.*]] = call ptr @omp_target_alloc(i64 8, i32 0)
// CHECK: %[[PTRTOINT:.*]] = ptrtoint ptr %[[ALLOC]] to i64
// CHECK: %[[INTTOPTR:.*]] = inttoptr i64 %[[PTRTOINT]] to ptr
// CHECK: call void @omp_target_free(ptr %[[INTTOPTR]], i32 0)
// CHECK: ret void
llvm.func @test_alloc_free_i64() -> () {
  %device = llvm.mlir.constant(0 : i32) : i32
  %1 = omp.target_allocmem %device : i32, i64
  omp.target_freemem %device, %1 : i32, i64
  llvm.return
}

// CHECK-LABEL: test_alloc_free_vector_1d_f32
// CHECK: %[[ALLOC:.*]] = call ptr @omp_target_alloc(i64 64, i32 0)
// CHECK: %[[PTRTOINT:.*]] = ptrtoint ptr %[[ALLOC]] to i64
// CHECK: %[[INTTOPTR:.*]] = inttoptr i64 %[[PTRTOINT]] to ptr
// CHECK: call void @omp_target_free(ptr %[[INTTOPTR]], i32 0)
// CHECK: ret void
llvm.func @test_alloc_free_vector_1d_f32() -> () {
  %device = llvm.mlir.constant(0 : i32) : i32
  %1 = omp.target_allocmem %device : i32, vector<16xf32>
  omp.target_freemem %device, %1 : i32, i64
  llvm.return
}

// CHECK-LABEL: test_alloc_free_vector_2d_f32
// CHECK: %[[ALLOC:.*]] = call ptr @omp_target_alloc(i64 1024, i32 0)
// CHECK: %[[PTRTOINT:.*]] = ptrtoint ptr %[[ALLOC]] to i64
// CHECK: %[[INTTOPTR:.*]] = inttoptr i64 %[[PTRTOINT]] to ptr
// CHECK: call void @omp_target_free(ptr %[[INTTOPTR]], i32 0)
// CHECK: ret void
llvm.func @test_alloc_free_vector_2d_f32() -> () {
  %device = llvm.mlir.constant(0 : i32) : i32
  %1 = omp.target_allocmem %device : i32, vector<16x16xf32>
  omp.target_freemem %device, %1 : i32, i64
  llvm.return
}
