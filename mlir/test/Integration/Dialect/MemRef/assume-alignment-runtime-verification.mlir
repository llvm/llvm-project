// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @main() {
  // This buffer is properly aligned. There should be no error.
  // CHECK-NOT: ^ memref is not aligned to 8
  %alloc = memref.alloca() : memref<5xf64>
  %0 = memref.assume_alignment %alloc, 8 : memref<5xf64>

  // Construct a memref descriptor with a pointer that is not aligned to 4.
  // This cannot be done with just the memref dialect. We have to resort to
  // the LLVM dialect.
  %c0 = llvm.mlir.constant(0 : index) : i64
  %c1 = llvm.mlir.constant(1 : index) : i64
  %c3 = llvm.mlir.constant(3 : index) : i64
  %unaligned_ptr = llvm.inttoptr %c3 : i64 to !llvm.ptr
  %4 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %5 = llvm.insertvalue %unaligned_ptr, %4[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %6 = llvm.insertvalue %unaligned_ptr, %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %8 = llvm.insertvalue %c0, %6[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %9 = llvm.insertvalue %c1, %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %10 = llvm.insertvalue %c1, %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %buffer = builtin.unrealized_conversion_cast %10 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: %[[ASSUME:.*]] = memref.assume_alignment %{{.*}}, 4 : memref<1xf32>
  // CHECK-NEXT: ^ memref is not aligned to 4
  // CHECK-NEXT: Location: loc({{.*}})
  %assume = memref.assume_alignment %buffer, 4 : memref<1xf32>

  return
}
