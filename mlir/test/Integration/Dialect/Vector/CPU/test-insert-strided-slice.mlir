// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() {
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %v1 = vector.broadcast %f1 : f32 to vector<4xf32>
  %v2 = vector.broadcast %f2 : f32 to vector<3xf32>
  %v3 = vector.broadcast %f3 : f32 to vector<4x4xf32>
  %v4 = vector.broadcast %f4 : f32 to vector<1xf32>

  %s1 = vector.insert_strided_slice %v1, %v3 {offsets = [2, 0], strides = [1]} : vector<4xf32> into vector<4x4xf32>
  %s2 = vector.insert_strided_slice %v2, %s1 {offsets = [1, 1], strides = [1]} : vector<3xf32> into vector<4x4xf32>
  %s3 = vector.insert_strided_slice %v2, %s2 {offsets = [0, 0], strides = [1]} : vector<3xf32> into vector<4x4xf32>
  %s4 = vector.insert_strided_slice %v4, %s3 {offsets = [3, 3], strides = [1]} : vector<1xf32> into vector<4x4xf32>

  vector.print %v3 : vector<4x4xf32>
  vector.print %s1 : vector<4x4xf32>
  vector.print %s2 : vector<4x4xf32>
  vector.print %s3 : vector<4x4xf32>
  vector.print %s4 : vector<4x4xf32>
  //
  // insert strided slice:
  //
  // CHECK: ( ( 3, 3, 3, 3 ), ( 3, 3, 3, 3 ), ( 3, 3, 3, 3 ), ( 3, 3, 3, 3 ) )
  // CHECK: ( ( 3, 3, 3, 3 ), ( 3, 3, 3, 3 ), ( 1, 1, 1, 1 ), ( 3, 3, 3, 3 ) )
  // CHECK: ( ( 3, 3, 3, 3 ), ( 3, 2, 2, 2 ), ( 1, 1, 1, 1 ), ( 3, 3, 3, 3 ) )
  // CHECK: ( ( 2, 2, 2, 3 ), ( 3, 2, 2, 2 ), ( 1, 1, 1, 1 ), ( 3, 3, 3, 3 ) )
  // CHECK: ( ( 2, 2, 2, 3 ), ( 3, 2, 2, 2 ), ( 1, 1, 1, 1 ), ( 3, 3, 3, 4 ) )

  return
}
