// RUN: mlir-opt %s -convert-vector-to-llvm="enable-x86vector" | mlir-opt | FileCheck %s

// CHECK-LABEL: func @avx512_mask_rndscale
func @avx512_mask_rndscale(%a: vector<16xf32>, %b: vector<8xf64>, %i32: i32, %i16: i16, %i8: i8)
  -> (vector<16xf32>, vector<8xf64>, vector<16xf32>, vector<8xf64>)
{
  // CHECK: x86vector.avx512.intr.mask.rndscale.ps.512
  %0 = x86vector.avx512.mask.rndscale %a, %i32, %a, %i16, %i32: vector<16xf32>
  // CHECK: x86vector.avx512.intr.mask.rndscale.pd.512
  %1 = x86vector.avx512.mask.rndscale %b, %i32, %b, %i8, %i32: vector<8xf64>

  // CHECK: x86vector.avx512.intr.mask.scalef.ps.512
  %2 = x86vector.avx512.mask.scalef %a, %a, %a, %i16, %i32: vector<16xf32>
  // CHECK: x86vector.avx512.intr.mask.scalef.pd.512
  %3 = x86vector.avx512.mask.scalef %b, %b, %b, %i8, %i32: vector<8xf64>

  // Keep results alive.
  return %0, %1, %2, %3 : vector<16xf32>, vector<8xf64>, vector<16xf32>, vector<8xf64>
}

// CHECK-LABEL: func @avx512_mask_compress
func @avx512_mask_compress(%k1: vector<16xi1>, %a1: vector<16xf32>,
                           %k2: vector<8xi1>, %a2: vector<8xi64>)
  -> (vector<16xf32>, vector<16xf32>, vector<8xi64>)
{
  // CHECK: x86vector.avx512.intr.mask.compress
  %0 = x86vector.avx512.mask.compress %k1, %a1 : vector<16xf32>
  // CHECK: x86vector.avx512.intr.mask.compress
  %1 = x86vector.avx512.mask.compress %k1, %a1 {constant_src = dense<5.0> : vector<16xf32>} : vector<16xf32>
  // CHECK: x86vector.avx512.intr.mask.compress
  %2 = x86vector.avx512.mask.compress %k2, %a2, %a2 : vector<8xi64>, vector<8xi64>
  return %0, %1, %2 : vector<16xf32>, vector<16xf32>, vector<8xi64>
}

// CHECK-LABEL: func @avx512_vp2intersect
func @avx512_vp2intersect(%a: vector<16xi32>, %b: vector<8xi64>)
  -> (vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>)
{
  // CHECK: x86vector.avx512.intr.vp2intersect.d.512
  %0, %1 = x86vector.avx512.vp2intersect %a, %a : vector<16xi32>
  // CHECK: x86vector.avx512.intr.vp2intersect.q.512
  %2, %3 = x86vector.avx512.vp2intersect %b, %b : vector<8xi64>
  return %0, %1, %2, %3 : vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>
}

// CHECK-LABEL: func @avx_rsqrt
func @avx_rsqrt(%a: vector<8xf32>) -> (vector<8xf32>)
{
  // CHECK: x86vector.avx.intr.rsqrt.ps.256
  %0 = x86vector.avx.rsqrt %a : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avx_dot
func @avx_dot(%a: vector<8xf32>, %b: vector<8xf32>) -> (vector<8xf32>)
{
  // CHECK: x86vector.avx.intr.dp.ps.256
  %0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
  return %0 : vector<8xf32>
}
