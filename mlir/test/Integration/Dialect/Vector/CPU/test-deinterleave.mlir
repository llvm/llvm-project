// RUN: mlir-opt %s -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN: -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %v0 = arith.constant dense<[1, 2, 3, 4]> : vector<4xi8>
  vector.print %v0 : vector<4xi8>
  // CHECK: ( 1, 2, 3, 4 )

  %v1, %v2 = vector.deinterleave %v0 : vector<4xi8> -> vector<2xi8>
  vector.print %v1 : vector<2xi8>
  vector.print %v2 : vector<2xi8>
  // CHECK: ( 1, 3 )
  // CHECK: ( 2, 4 )

  return
}
