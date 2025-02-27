// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %c0 = arith.constant dense<[0, 1, 2, 3]>: vector<4xindex>
  %c1 = arith.constant dense<[0, 1]>: vector<2xindex>
  %c2 = arith.constant 2 : index

  %v1 = vector.broadcast %c0 : vector<4xindex> to vector<2x4xindex>
  %v2 = vector.broadcast %c1 : vector<2xindex> to vector<4x2xindex>
  %v3 = vector.transpose %v2, [1, 0] : vector<4x2xindex> to vector<2x4xindex>
  %v4 = vector.broadcast %c2 : index to vector<2x4xindex>

  %v5 = arith.addi %v1, %v3 : vector<2x4xindex>

  vector.print %v1 : vector<2x4xindex>
  vector.print %v3 : vector<2x4xindex>
  vector.print %v4 : vector<2x4xindex>
  vector.print %v5 : vector<2x4xindex>

  //
  // created index vectors:
  //
  // CHECK: ( ( 0, 1, 2, 3 ), ( 0, 1, 2, 3 ) )
  // CHECK: ( ( 0, 0, 0, 0 ), ( 1, 1, 1, 1 ) )
  // CHECK: ( ( 2, 2, 2, 2 ), ( 2, 2, 2, 2 ) )
  // CHECK: ( ( 0, 1, 2, 3 ), ( 1, 2, 3, 4 ) )

  return
}
