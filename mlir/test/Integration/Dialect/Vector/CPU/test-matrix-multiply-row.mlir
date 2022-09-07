// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -O0 -enable-matrix -matrix-allow-contract -matrix-default-layout=row-major \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() {
  %f0 = arith.constant 0.0: f64
  %f1 = arith.constant 1.0: f64
  %f2 = arith.constant 2.0: f64
  %f3 = arith.constant 3.0: f64
  %f4 = arith.constant 4.0: f64
  %f5 = arith.constant 5.0: f64
  %f6 = arith.constant 6.0: f64
  %f7 = arith.constant 7.0: f64

  // Construct test vectors.
  %0 = vector.broadcast %f0 : f64 to vector<4xf64>
  %1 = vector.insert %f1, %0[1] : f64 into vector<4xf64>
  %2 = vector.insert %f2, %1[2] : f64 into vector<4xf64>
  %a = vector.insert %f3, %2[3] : f64 into vector<4xf64>
  %3 = vector.broadcast %f4 : f64 to vector<4xf64>
  %4 = vector.insert %f5, %3[1] : f64 into vector<4xf64>
  %5 = vector.insert %f6, %4[2] : f64 into vector<4xf64>
  %b = vector.insert %f7, %5[3] : f64 into vector<4xf64>

  vector.print %a : vector<4xf64>
  vector.print %b : vector<4xf64>
  //
  // test vectors:
  //
  // CHECK: ( 0, 1, 2, 3 )
  // CHECK: ( 4, 5, 6, 7 )

  // Performs matrix x matrix, interpreting the vectors as
  // flattened row-major 2-D matrices.
  //
  // ( 0, 1 )     (4, 5)     (  6,  7 )
  //           x          =
  // ( 2, 3 )     (6, 7)     ( 26, 31 )
  //
  %c = vector.matrix_multiply %a, %b
      { lhs_rows = 2: i32, lhs_columns = 2: i32 , rhs_columns = 2: i32 }
      : (vector<4xf64>, vector<4xf64>) -> vector<4xf64>

  vector.print %c : vector<4xf64>
  //
  // matrix x matrix:
  //
  // CHECK: ( 6, 7, 26, 31 )

  return
}
