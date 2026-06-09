// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @store_dynamic(%memref: memref<?xf32>, %index: index) {
  %cst = arith.constant 1.0 : f32
  memref.store %cst, %memref[%index] :  memref<?xf32>
  return
}

func.func @main() {
  // Allocate a memref<10xf32>, but disguise it as a memref<5xf32>. This is
  // necessary because "-test-cf-assert" does not abort the program and we do
  // not want to segfault when running the test case.
  %alloc = memref.alloca() : memref<10xf32>
  %ptr = memref.extract_aligned_pointer_as_index %alloc : memref<10xf32> -> index
  %ptr_i64 = arith.index_cast %ptr : index to i64
  %ptr_llvm = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr
  %c0 = llvm.mlir.constant(0 : index) : i64
  %c1 = llvm.mlir.constant(1 : index) : i64
  %c5 = llvm.mlir.constant(5 : index) : i64
  %4 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %5 = llvm.insertvalue %ptr_llvm, %4[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %6 = llvm.insertvalue %ptr_llvm, %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %8 = llvm.insertvalue %c0, %6[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %9 = llvm.insertvalue %c5, %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %10 = llvm.insertvalue %c1, %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %buffer = builtin.unrealized_conversion_cast %10 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<5xf32>
  %cast = memref.cast %buffer : memref<5xf32> to memref<?xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32>
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  %c9 = arith.constant 9 : index
  func.call @store_dynamic(%cast, %c9) : (memref<?xf32>, index) -> ()

  return
}

