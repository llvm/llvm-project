// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm="enable-x86vector" -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx" --dlopen=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() -> i32 {
  %i0 = arith.constant 0 : i32
  %i4 = arith.constant 4 : i32

  %a = arith.constant dense<[1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0]> : vector<8xf32>
  %b = arith.constant dense<[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : vector<8xf32>
  %r = x86vector.avx.intr.dot %a, %b : vector<8xf32>

  %1 = vector.extractelement %r[%i0 : i32]: vector<8xf32>
  %2 = vector.extractelement %r[%i4 : i32]: vector<8xf32>
  %d = arith.addf %1, %2 : f32

  // CHECK: ( 110, 110, 110, 110, 382, 382, 382, 382 )
  // CHECK: 492
  vector.print %r : vector<8xf32>
  vector.print %d : f32

  return %i0 : i32
}
