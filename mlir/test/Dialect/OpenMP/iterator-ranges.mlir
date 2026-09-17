// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s --canonicalize --verify-each | FileCheck %s

// Empty ranges yield empty lists, for either endpoint convention.
// CHECK-LABEL: func.func @empty_ranges
func.func @empty_ranges() -> (!omp.iterated<index>, !omp.iterated<index>) {
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %neg = arith.constant -1 : index
  // CHECK: omp.iterator
  // CHECK: } inclusive -> !omp.iterated<index>
  %a = omp.iterator(%i: index) = (%two to %one step %one) {
    omp.yield(%i : index)
  } inclusive -> !omp.iterated<index>
  // CHECK: omp.iterator
  // CHECK: } -> !omp.iterated<index>
  %b = omp.iterator(%i: index) = (%one to %two step %neg) {
    omp.yield(%i : index)
  } -> !omp.iterated<index>
  return %a, %b : !omp.iterated<index>, !omp.iterated<index>
}

// CHECK-LABEL: func.func @wide_range
func.func @wide_range(%hi: i128, %step: i128) -> !omp.iterated<i128> {
  %zero = arith.constant 0 : i128
  // CHECK: omp.iterator(%{{.*}}: i128) =
  // CHECK: omp.yield(%{{.*}} : i128)
  // CHECK: } inclusive -> !omp.iterated<i128>
  %a = omp.iterator(%i: i128) = (%zero to %hi step %step) {
    omp.yield(%i : i128)
  } inclusive -> !omp.iterated<i128>
  return %a : !omp.iterated<i128>
}
