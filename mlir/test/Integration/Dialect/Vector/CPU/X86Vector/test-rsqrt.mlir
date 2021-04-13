// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm="enable-x86vector" -convert-std-to-llvm | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: %lli --jit-kind=mcjit --entry-function=entry --mattr="avx512bw" --dlopen=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
// TODO: drop lli's --jit-kind flag once PR#49906 (https://bugs.llvm.org/show_bug.cgi?id=49906) is fixed.

func @entry() -> i32 {
  %i0 = constant 0 : i32

  %v = std.constant dense<[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]> : vector<8xf32>
  %r = x86vector.avx.rsqrt %v : vector<8xf32>
  // CHECK: ( 2.82764, 1.99951, 1.41382, 0.999756, 0.706909, 0.499878, 0.353455, 0.249939 )
  vector.print %r : vector<8xf32>

  return %i0 : i32
}
