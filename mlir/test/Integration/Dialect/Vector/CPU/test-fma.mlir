// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %f1 = arith.constant 1.0: f32
  %f3 = arith.constant 3.0: f32
  %f7 = arith.constant 7.0: f32
  %v1 = vector.broadcast %f1 : f32 to vector<8xf32>
  %v3 = vector.broadcast %f3 : f32 to vector<8xf32>
  %v7 = vector.broadcast %f7 : f32 to vector<8xf32>
  vector.print %v1 : vector<8xf32>
  vector.print %v3 : vector<8xf32>
  vector.print %v7 : vector<8xf32>
  //
  // test vectors:
  //
  // CHECK: ( 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK: ( 3, 3, 3, 3, 3, 3, 3, 3 )
  // CHECK: ( 7, 7, 7, 7, 7, 7, 7, 7 )

  %v = vector.fma %v3, %v7, %v1: vector<8xf32>
  vector.print %v : vector<8xf32>
  // CHECK: ( 22, 22, 22, 22, 22, 22, 22, 22 )

  return
}
