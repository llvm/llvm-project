// RUN: mlir-opt %s -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %v1 = vector.splat %f1 : vector<2x4xf32>
  %v2 = vector.splat %f2 : vector<2x4xf32>
  vector.print %v1 : vector<2x4xf32>
  vector.print %v2 : vector<2x4xf32>
  //
  // Test vectors:
  //
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ) )
  // CHECK: ( ( 2, 2, 2, 2 ), ( 2, 2, 2, 2 ) )

  %v3 = vector.interleave %v1, %v2 : vector<2x4xf32>
  vector.print %v3 : vector<2x8xf32>
  // CHECK: ( ( 1, 2, 1, 2, 1, 2, 1, 2 ), ( 1, 2, 1, 2, 1, 2, 1, 2 ) )

  return
}
