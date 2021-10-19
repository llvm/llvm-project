// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32
  %f6 = arith.constant 6.0: f32

  // Construct test vector.
  %0 = vector.broadcast %f1 : f32 to vector<3x2xf32>
  %1 = vector.insert %f2, %0[0, 1] : f32 into vector<3x2xf32>
  %2 = vector.insert %f3, %1[1, 0] : f32 into vector<3x2xf32>
  %3 = vector.insert %f4, %2[1, 1] : f32 into vector<3x2xf32>
  %4 = vector.insert %f5, %3[2, 0] : f32 into vector<3x2xf32>
  %x = vector.insert %f6, %4[2, 1] : f32 into vector<3x2xf32>
  vector.print %x : vector<3x2xf32>
  // CHECK:  ( ( 1, 2 ), ( 3, 4 ), ( 5, 6 ) )

  // Reshapes.
  %a = vector.shape_cast %x : vector<3x2xf32> to vector<3x2xf32>
  %b = vector.shape_cast %x : vector<3x2xf32> to vector<2x3xf32>
  %c = vector.shape_cast %x : vector<3x2xf32> to vector<6xf32>
  %d = vector.shape_cast %c : vector<6xf32> to vector<2x3xf32>
  %e = vector.shape_cast %c : vector<6xf32> to vector<3x2xf32>

  // Reshaped vectors:
  // CHECK:  ( ( 1, 2 ), ( 3, 4 ), ( 5, 6 ) )
  // CHECK: ( ( 1, 2, 3 ), ( 4, 5, 6 ) )
  // CHECK: ( 1, 2, 3, 4, 5, 6 )
  // CHECK: ( ( 1, 2, 3 ), ( 4, 5, 6 ) )
  // CHECK: ( ( 1, 2 ), ( 3, 4 ), ( 5, 6 ) )
  vector.print %a : vector<3x2xf32>
  vector.print %b : vector<2x3xf32>
  vector.print %c : vector<6xf32>
  vector.print %d : vector<2x3xf32>
  vector.print %e : vector<3x2xf32>

  return
}
