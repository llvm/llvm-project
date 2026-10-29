// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --canonicalize | FileCheck %s

// A range pointing away from its endpoint is valid and produces no entries,
// for either endpoint convention.
// CHECK-LABEL: func.func @empty_ranges
func.func @empty_ranges(%x: !llvm.ptr)
    -> (!omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>,
        !omp.iterated<!llvm.ptr>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %neg = arith.constant -1 : index
  // CHECK: omp.iterator
  // CHECK: } -> !omp.iterated<!llvm.ptr>
  %positive = omp.iterator(%i: index) = (%c3 to %c1 step %c1) {
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  // CHECK: omp.iterator
  // CHECK: } -> !omp.iterated<!llvm.ptr>
  %negative = omp.iterator(%i: index) = (%c1 to %c3 step %neg) {
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  // CHECK: omp.iterator
  // CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
  %inclusive = omp.iterator(%i: index) = (%c3 to %c1 step %c1) {
    omp.yield(%x : !llvm.ptr)
  } inclusive -> !omp.iterated<!llvm.ptr>
  return %positive, %negative, %inclusive
      : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>,
        !omp.iterated<!llvm.ptr>
}
