// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() {
  %i = arith.constant 2147483647: i32
  %l = arith.constant 9223372036854775807 : i64

  %f0 = arith.constant 0.0: f32
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32

  // Test simple broadcasts.
  %vi = vector.broadcast %i : i32 to vector<2xi32>
  %vl = vector.broadcast %l : i64 to vector<2xi64>
  %vf = vector.broadcast %f1 : f32 to vector<2x2x2xf32>
  vector.print %vi : vector<2xi32>
  vector.print %vl : vector<2xi64>
  vector.print %vf : vector<2x2x2xf32>
  // CHECK: ( 2147483647, 2147483647 )
  // CHECK: ( 9223372036854775807, 9223372036854775807 )
  // CHECK: ( ( ( 1, 1 ), ( 1, 1 ) ), ( ( 1, 1 ), ( 1, 1 ) ) )

  // Test "duplication" in leading dimensions.
  %v0 = vector.broadcast %f1 : f32 to vector<4xf32>
  %v1 = vector.insert %f2, %v0[1] : f32 into vector<4xf32>
  %v2 = vector.insert %f3, %v1[2] : f32 into vector<4xf32>
  %v3 = vector.insert %f4, %v2[3] : f32 into vector<4xf32>
  %v4 = vector.broadcast %v3 : vector<4xf32> to vector<3x4xf32>
  %v5 = vector.broadcast %v3 : vector<4xf32> to vector<2x2x4xf32>
  vector.print %v3 : vector<4xf32>
  vector.print %v4 : vector<3x4xf32>
  vector.print %v5 : vector<2x2x4xf32>
  // CHECK: ( 1, 2, 3, 4 )
  // CHECK: ( ( 1, 2, 3, 4 ), ( 1, 2, 3, 4 ), ( 1, 2, 3, 4 ) )
  // CHECK: ( ( ( 1, 2, 3, 4 ), ( 1, 2, 3, 4 ) ), ( ( 1, 2, 3, 4 ), ( 1, 2, 3, 4 ) ) )

  // Test straightforward "stretch" on a 1-D "scalar".
  %x = vector.broadcast %f5 : f32 to vector<1xf32>
  %y = vector.broadcast %x  : vector<1xf32> to vector<8xf32>
  vector.print %y : vector<8xf32>
  // CHECK : ( 5, 5, 5, 5, 5, 5, 5, 5 )

  // Test "stretch" in leading dimension.
  %s = vector.broadcast %v3 : vector<4xf32> to vector<1x4xf32>
  %t = vector.broadcast %s  : vector<1x4xf32> to vector<3x4xf32>
  vector.print %s : vector<1x4xf32>
  vector.print %t : vector<3x4xf32>
  // CHECK: ( ( 1, 2, 3, 4 ) )
  // CHECK: ( ( 1, 2, 3, 4 ), ( 1, 2, 3, 4 ), ( 1, 2, 3, 4 ) )

  // Test "stretch" in trailing dimension.
  %a0 = vector.broadcast %f1 : f32 to vector<3x1xf32>
  %a1 = vector.insert %f2, %a0[1, 0] : f32 into vector<3x1xf32>
  %a2 = vector.insert %f3, %a1[2, 0] : f32 into vector<3x1xf32>
  %a3 = vector.broadcast %a2 : vector<3x1xf32> to vector<3x4xf32>
  vector.print %a2 : vector<3x1xf32>
  vector.print %a3 : vector<3x4xf32>
  // CHECK: ( ( 1 ), ( 2 ), ( 3 ) )
  // CHECK: ( ( 1, 1, 1, 1 ), ( 2, 2, 2, 2 ), ( 3, 3, 3, 3 ) )

  // Test "stretch" in middle dimension.
  %m0 = vector.broadcast %f0 : f32 to vector<3x1x2xf32>
  %m1 = vector.insert %f1, %m0[0, 0, 1] : f32 into vector<3x1x2xf32>
  %m2 = vector.insert %f2, %m1[1, 0, 0] : f32 into vector<3x1x2xf32>
  %m3 = vector.insert %f3, %m2[1, 0, 1] : f32 into vector<3x1x2xf32>
  %m4 = vector.insert %f4, %m3[2, 0, 0] : f32 into vector<3x1x2xf32>
  %m5 = vector.insert %f5, %m4[2, 0, 1] : f32 into vector<3x1x2xf32>
  %m6 = vector.broadcast %m5 : vector<3x1x2xf32> to vector<3x4x2xf32>
  vector.print %m5 : vector<3x1x2xf32>
  vector.print %m6 : vector<3x4x2xf32>
  // CHECK: ( ( ( 0, 1 ) ), ( ( 2, 3 ) ), ( ( 4, 5 ) ) )
  // CHECK: ( ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME: ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ),
  // CHECK-SAME: ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ) )

  return
}
