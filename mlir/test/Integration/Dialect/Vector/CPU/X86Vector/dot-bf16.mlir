// RUN: mlir-opt %s -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm="enable-x86vector" -convert-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx512bf16" --dlopen=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() -> i32 {
  %i0 = arith.constant 0 : i32
  %i3 = arith.constant 3 : i32

  %src = arith.constant dense<1.0> : vector<4xf32>
  %a = arith.constant dense<[1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0]> : vector<8xbf16>
  %b = arith.constant dense<[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : vector<8xbf16>
  %dst = x86vector.avx512bf16.dot %src, %a, %b : vector<8xbf16> -> vector<4xf32>

  %1 = vector.extractelement %dst[%i0 : i32] : vector<4xf32>
  %2 = vector.extractelement %dst[%i3 : i32] : vector<4xf32>
  %d = arith.addf %1, %2 : f32

  // CHECK: ( 30, 82, 150, 234 )
  // CHECK: 264
  vector.print %dst : vector<4xf32>
  vector.print %d : f32

  return %i0 : i32
}
