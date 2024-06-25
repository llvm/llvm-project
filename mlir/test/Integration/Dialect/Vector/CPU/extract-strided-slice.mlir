// RUN: mlir-opt %s -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %f0 = arith.constant 0.0: f32
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %v1 = vector.broadcast %f1 : f32 to vector<8xf32>
  %v2 = vector.broadcast %f2 : f32 to vector<8xf32>
  %v3 = vector.broadcast %f3 : f32 to vector<8xf32>
  %v4 = vector.broadcast %f4 : f32 to vector<8xf32>

  %a0 = vector.broadcast %f0 : f32 to vector<4x4x8xf32>
  %a1 = vector.insert %v1, %a0[1, 1] : vector<8xf32> into vector<4x4x8xf32>
  %a2 = vector.insert %v2, %a1[1, 2] : vector<8xf32> into vector<4x4x8xf32>
  %a3 = vector.insert %v3, %a2[2, 1] : vector<8xf32> into vector<4x4x8xf32>
  %a4 = vector.insert %v4, %a3[2, 2] : vector<8xf32> into vector<4x4x8xf32>

  %ss = vector.extract_strided_slice %a4 {offsets = [1, 1], sizes = [2, 2], strides = [1, 1]} : vector<4x4x8xf32> to vector<2x2x8xf32>

  vector.print %ss : vector<2x2x8xf32>
  //
  // extract strided slice:
  //
  // CHECK: ( ( ( 1, 1, 1, 1, 1, 1, 1, 1 ), ( 2, 2, 2, 2, 2, 2, 2, 2 ) ), ( ( 3, 3, 3, 3, 3, 3, 3, 3 ), ( 4, 4, 4, 4, 4, 4, 4, 4 ) ) )

  return
}
