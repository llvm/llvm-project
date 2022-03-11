// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %0 = vector.constant_mask [4] : vector<8xi1>
  vector.print %0 : vector<8xi1>
  // CHECK: ( 1, 1, 1, 1, 0, 0, 0, 0 )

  %1 = vector.constant_mask [1, 3] : vector<4x4xi1>
  vector.print %1 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )

  %2 = vector.constant_mask [2, 2] : vector<4x4xi1>
  vector.print %2 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )

  %3 = vector.constant_mask [2, 4] : vector<4x4xi1>
  vector.print %3 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )

  %4 = vector.constant_mask [3, 1] : vector<4x4xi1>
  vector.print %4 : vector<4x4xi1>
  // CHECK: ( ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 1, 0, 0, 0 ), ( 0, 0, 0, 0 ) )

  %5 = vector.constant_mask [3, 2] : vector<4x4xi1>
  vector.print %5 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 1, 1, 0, 0 ), ( 0, 0, 0, 0 ) )

  %6 = vector.constant_mask [4, 3] : vector<4x4xi1>
  vector.print %6 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ) )

  %7 = vector.constant_mask [4, 4] : vector<4x4xi1>
  vector.print %7 : vector<4x4xi1>
  // CHECK: ( ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ), ( 1, 1, 1, 1 ) )

  %8 = vector.constant_mask [1, 2, 3] : vector<2x3x4xi1>
  vector.print %8 : vector<2x3x4xi1>
  // CHECK: ( ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ) ), ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) ) )

  %9 = vector.constant_mask [2, 2, 3] : vector<2x3x4xi1>
  vector.print %9 : vector<2x3x4xi1>
  // CHECK: ( ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ) ), ( ( 1, 1, 1, 0 ), ( 1, 1, 1, 0 ), ( 0, 0, 0, 0 ) ) )

  return
}

