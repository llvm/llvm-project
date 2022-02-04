// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %f0 = arith.constant 0.0: f32
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32
  %f6 = arith.constant 6.0: f32
  %f7 = arith.constant 7.0: f32
  %f8 = arith.constant 8.0: f32

  // Construct test vectors and matrices.
  %0 = vector.broadcast %f1 : f32 to vector<2xf32>
  %a = vector.insert %f2, %0[1] : f32 into vector<2xf32>
  %1 = vector.broadcast %f3 : f32 to vector<2xf32>
  %b = vector.insert %f4, %1[1] : f32 into vector<2xf32>
  %2 = vector.broadcast %f5 : f32 to vector<2xf32>
  %c = vector.insert %f6, %2[1] : f32 into vector<2xf32>
  %3 = vector.broadcast %f7 : f32 to vector<2xf32>
  %d = vector.insert %f8, %3[1] : f32 into vector<2xf32>
  %4 = vector.broadcast %f0 : f32 to vector<2x2xf32>
  %5 = vector.insert %a, %4[0] : vector<2xf32> into vector<2x2xf32>
  %A = vector.insert %b, %5[1] : vector<2xf32> into vector<2x2xf32>
  %6 = vector.broadcast %f0 : f32 to vector<2x2xf32>
  %7 = vector.insert %c, %6[0] : vector<2xf32> into vector<2x2xf32>
  %B = vector.insert %d, %7[1] : vector<2xf32> into vector<2x2xf32>
  %8 = vector.broadcast %f0 : f32 to vector<3x2xf32>
  %9 = vector.insert %a, %8[0] : vector<2xf32> into vector<3x2xf32>
  %10 = vector.insert %b, %9[1] : vector<2xf32> into vector<3x2xf32>
  %C = vector.insert %c, %10[2] : vector<2xf32> into vector<3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<2x4xf32>
  %11 = vector.insert_strided_slice %A, %cst {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<2x4xf32>
  %D = vector.insert_strided_slice %B, %11 {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<2x4xf32>

  vector.print %A : vector<2x2xf32>
  vector.print %B : vector<2x2xf32>
  vector.print %C : vector<3x2xf32>
  vector.print %D : vector<2x4xf32>
  //
  // test matrices:
  //
  // CHECK: ( ( 1, 2 ), ( 3, 4 ) )
  // CHECK: ( ( 5, 6 ), ( 7, 8 ) )
  // CHECK: ( ( 1, 2 ), ( 3, 4 ), ( 5, 6 ) )
  // CHECK: ( ( 1, 2, 5, 6 ), ( 3, 4, 7, 8 ) )

  %tA = vector.transpose %A, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
  %tB = vector.transpose %B, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
  %tC = vector.transpose %C, [1, 0] : vector<3x2xf32> to vector<2x3xf32>
  %tD = vector.transpose %D, [1, 0] : vector<2x4xf32> to vector<4x2xf32>

  vector.print %tA : vector<2x2xf32>
  vector.print %tB : vector<2x2xf32>
  vector.print %tC : vector<2x3xf32>
  vector.print %tD : vector<4x2xf32>
  //
  // transposed matrices:
  //
  // CHECK: ( ( 1, 3 ), ( 2, 4 ) )
  // CHECK: ( ( 5, 7 ), ( 6, 8 ) )
  // CHECK: ( ( 1, 3, 5 ), ( 2, 4, 6 ) )
  // CHECK: ( ( 1, 3 ), ( 2, 4 ), ( 5, 7 ), ( 6, 8 ) )

  %idD = vector.transpose %D, [0, 1] : vector<2x4xf32> to vector<2x4xf32>
  %ttD = vector.transpose %tD, [1, 0] : vector<4x2xf32> to vector<2x4xf32>

  vector.print %idD : vector<2x4xf32>
  vector.print %ttD : vector<2x4xf32>
  //
  // back to original after transpose matrices:
  //
  // CHECK: ( ( 1, 2, 5, 6 ), ( 3, 4, 7, 8 ) )
  // CHECK: ( ( 1, 2, 5, 6 ), ( 3, 4, 7, 8 ) )

  // Construct test tensor.
  %p = vector.broadcast %f1 : f32 to vector<2x2x2xf32>
  %q = vector.insert %f2, %p[0, 0, 1] : f32 into vector<2x2x2xf32>
  %r = vector.insert %f3, %q[0, 1, 0] : f32 into vector<2x2x2xf32>
  %s = vector.insert %f4, %r[0, 1, 1] : f32 into vector<2x2x2xf32>
  %t = vector.insert %f5, %s[1, 0, 0] : f32 into vector<2x2x2xf32>
  %u = vector.insert %f6, %t[1, 0, 1] : f32 into vector<2x2x2xf32>
  %v = vector.insert %f7, %u[1, 1, 0] : f32 into vector<2x2x2xf32>
  %w = vector.insert %f8, %v[1, 1, 1] : f32 into vector<2x2x2xf32>

  vector.print %w : vector<2x2x2xf32>
  //
  // test tensors:
  //
  // CHECK: ( ( ( 1, 2 ), ( 3, 4 ) ), ( ( 5, 6 ), ( 7, 8 ) ) )

  %tP = vector.transpose %w, [0, 1, 2] : vector<2x2x2xf32> to vector<2x2x2xf32>
  %tQ = vector.transpose %w, [0, 2, 1] : vector<2x2x2xf32> to vector<2x2x2xf32>
  %tR = vector.transpose %w, [1, 0, 2] : vector<2x2x2xf32> to vector<2x2x2xf32>
  %tS = vector.transpose %w, [2, 0, 1] : vector<2x2x2xf32> to vector<2x2x2xf32>
  %tT = vector.transpose %w, [1, 2, 0] : vector<2x2x2xf32> to vector<2x2x2xf32>
  %tU = vector.transpose %w, [2, 1, 0] : vector<2x2x2xf32> to vector<2x2x2xf32>

  vector.print %tP : vector<2x2x2xf32>
  vector.print %tQ : vector<2x2x2xf32>
  vector.print %tR : vector<2x2x2xf32>
  vector.print %tS : vector<2x2x2xf32>
  vector.print %tT : vector<2x2x2xf32>
  vector.print %tU : vector<2x2x2xf32>
  //
  // transposed tensors:
  //
  // CHECK: ( ( ( 1, 2 ), ( 3, 4 ) ), ( ( 5, 6 ), ( 7, 8 ) ) )
  // CHECK: ( ( ( 1, 3 ), ( 2, 4 ) ), ( ( 5, 7 ), ( 6, 8 ) ) )
  // CHECK: ( ( ( 1, 2 ), ( 5, 6 ) ), ( ( 3, 4 ), ( 7, 8 ) ) )
  // CHECK: ( ( ( 1, 3 ), ( 5, 7 ) ), ( ( 2, 4 ), ( 6, 8 ) ) )
  // CHECK: ( ( ( 1, 5 ), ( 2, 6 ) ), ( ( 3, 7 ), ( 4, 8 ) ) )
  // CHECK: ( ( ( 1, 5 ), ( 3, 7 ) ), ( ( 2, 6 ), ( 4, 8 ) ) )

  return
}
