// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %f0 = constant 0.0: f32
  %f1 = constant 1.0: f32
  %f2 = constant 2.0: f32
  %f3 = constant 3.0: f32
  %f4 = constant 4.0: f32
  %f5 = constant 5.0: f32
  %f6 = constant 6.0: f32
  %f7 = constant 7.0: f32
  %f8 = constant 8.0: f32
  %f9 = constant 9.0: f32
  %f10 = constant 10.0: f32
  %f11 = constant 11.0: f32
  %f12 = constant 12.0: f32
  %f13 = constant 13.0: f32
  %f14 = constant 14.0: f32
  %f15 = constant 15.0: f32

  %a0 = vector.broadcast %f0 : f32 to vector<3x3xf32>
  %a1 = vector.insert %f0, %a0[0, 0] : f32 into vector<3x3xf32>
  %a2 = vector.insert %f1, %a1[0, 1] : f32 into vector<3x3xf32>
  %a3 = vector.insert %f2, %a2[0, 2] : f32 into vector<3x3xf32>
  %a4 = vector.insert %f4, %a3[1, 0] : f32 into vector<3x3xf32>
  %a5 = vector.insert %f5, %a4[1, 1] : f32 into vector<3x3xf32>
  %a6 = vector.insert %f6, %a5[1, 2] : f32 into vector<3x3xf32>
  %a7 = vector.insert %f8, %a6[2, 0] : f32 into vector<3x3xf32>
  %a8 = vector.insert %f9, %a7[2, 1] : f32 into vector<3x3xf32>
  %a9 = vector.insert %f10, %a8[2, 2] : f32 into vector<3x3xf32>

  %b0 = vector.broadcast %f0 : f32 to vector<3x1xf32>
  %b1 = vector.insert %f3, %b0[0, 0] : f32 into vector<3x1xf32>
  %b2 = vector.insert %f7, %b1[1, 0] : f32 into vector<3x1xf32>
  %b3 = vector.insert %f11, %b2[2, 0] : f32 into vector<3x1xf32>

  %c0 = vector.broadcast %f0 : f32 to vector<1x3xf32>
  %c1 = vector.insert %f12, %c0[0, 0] : f32 into vector<1x3xf32>
  %c2 = vector.insert %f13, %c1[0, 1] : f32 into vector<1x3xf32>
  %c3 = vector.insert %f14, %c2[0, 2] : f32 into vector<1x3xf32>

  %d0 = vector.broadcast %f0 : f32 to vector<1x1xf32>
  %d1 = vector.insert %f15, %d0[0, 0] : f32 into vector<1x1xf32>

  vector.print %a9 : vector<3x3xf32>
  vector.print %b3 : vector<3x1xf32>
  vector.print %c3 : vector<1x3xf32>
  vector.print %d1 : vector<1x1xf32>
  //
  // input slices:
  //
  // CHECK: ( ( 0, 1, 2 ), ( 4, 5, 6 ), ( 8, 9, 10 ) )
  // CHECK: ( ( 3 ), ( 7 ), ( 11 ) )
  // CHECK: ( ( 12, 13, 14 ) )
  // CHECK: ( ( 15 ) )

  %vt = vector.tuple %a9, %b3, %c3, %d1 :
     vector<3x3xf32>, vector<3x1xf32>, vector<1x3xf32>, vector<1x1xf32>
  %is = vector.insert_slices %vt, [3, 3], [1, 1] :
     tuple<vector<3x3xf32>, vector<3x1xf32>, vector<1x3xf32>, vector<1x1xf32>> into vector<4x4xf32>

  vector.print %is : vector<4x4xf32>
  //
  // insert slices:
  //
  // CHECK: ( ( 0, 1, 2, 3 ), ( 4, 5, 6, 7 ), ( 8, 9, 10, 11 ), ( 12, 13, 14, 15 ) )

  return
}
