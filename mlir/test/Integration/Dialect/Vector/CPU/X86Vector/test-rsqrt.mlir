// RUN: mlir-opt %s -convert-vector-to-llvm="enable-x86vector" -convert-func-to-llvm | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx" --dlopen=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() -> i32 {
  %i0 = arith.constant 0 : i32

  %v = arith.constant dense<[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]> : vector<8xf32>
  %r = x86vector.avx.rsqrt %v : vector<8xf32>
  // `rsqrt` may produce slightly different results on Intel and AMD machines: accept both results here.
  // CHECK: {{( 2.82[0-9]*, 1.99[0-9]*, 1.41[0-9]*, 0.99[0-9]*, 0.70[0-9]*, 0.49[0-9]*, 0.35[0-9]*, 0.24[0-9]* )}}
  vector.print %r : vector<8xf32>

  return %i0 : i32
}
