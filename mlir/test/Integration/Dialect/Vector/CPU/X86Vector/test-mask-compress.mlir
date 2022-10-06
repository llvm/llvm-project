// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm="enable-x86vector" -convert-func-to-llvm -reconcile-unrealized-casts  | \
// RUN: mlir-translate  --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx512bw" --dlopen=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() -> i32 {
  %i0 = arith.constant 0 : i32

  %a = arith.constant dense<[1., 0., 0., 2., 4., 3., 5., 7., 8., 1., 5., 5., 3., 1., 0., 7.]> : vector<16xf32>
  %k = arith.constant dense<[1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0]> : vector<16xi1>
  %r1 = x86vector.avx512.mask.compress %k, %a : vector<16xf32>
  %r2 = x86vector.avx512.mask.compress %k, %a {constant_src = dense<5.0> : vector<16xf32>} : vector<16xf32>

  vector.print %r1 : vector<16xf32>
  // CHECK: ( 1, 0, 2, 4, 5, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0 )

  vector.print %r2 : vector<16xf32>
  // CHECK: ( 1, 0, 2, 4, 5, 5, 3, 1, 0, 5, 5, 5, 5, 5, 5, 5 )

  %src = arith.constant dense<[0., 2., 1., 8., 6., 4., 4., 3., 2., 8., 5., 6., 3., 7., 6., 9.]> : vector<16xf32>
  %r3 = x86vector.avx512.mask.compress %k, %a, %src : vector<16xf32>, vector<16xf32>

  vector.print %r3 : vector<16xf32>
  // CHECK: ( 1, 0, 2, 4, 5, 5, 3, 1, 0, 8, 5, 6, 3, 7, 6, 9 )

  return %i0 : i32
}
