// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

func @avx512_mask_rndscale(%a: vector<16xf32>, %b: vector<8xf64>, %i32: i32, %i16: i16, %i8: i8)
  -> (vector<16xf32>, vector<8xf64>)
{
  // CHECK: avx512.mask.rndscale {{.*}}: vector<16xf32>
  %0 = avx512.mask.rndscale %a, %i32, %a, %i16, %i32 : vector<16xf32>
  // CHECK: avx512.mask.rndscale {{.*}}: vector<8xf64>
  %1 = avx512.mask.rndscale %b, %i32, %b, %i8, %i32 : vector<8xf64>
  return %0, %1: vector<16xf32>, vector<8xf64>
}

func @avx512_scalef(%a: vector<16xf32>, %b: vector<8xf64>, %i32: i32, %i16: i16, %i8: i8)
  -> (vector<16xf32>, vector<8xf64>)
{
  // CHECK: avx512.mask.scalef {{.*}}: vector<16xf32>
  %0 = avx512.mask.scalef %a, %a, %a, %i16, %i32: vector<16xf32>
  // CHECK: avx512.mask.scalef {{.*}}: vector<8xf64>
  %1 = avx512.mask.scalef %b, %b, %b, %i8, %i32 : vector<8xf64>
  return %0, %1: vector<16xf32>, vector<8xf64>
}

func @avx512_vp2intersect(%a: vector<16xi32>, %b: vector<8xi64>)
  -> (vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>)
{
  // CHECK: avx512.vp2intersect {{.*}} : vector<16xi32>
  %0, %1 = avx512.vp2intersect %a, %a : vector<16xi32>
  // CHECK: avx512.vp2intersect {{.*}} : vector<8xi64>
  %2, %3 = avx512.vp2intersect %b, %b : vector<8xi64>
  return %0, %1, %2, %3 : vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>
}

func @avx512_mask_compress(%k1: vector<16xi1>, %a1: vector<16xf32>,
                           %k2: vector<8xi1>, %a2: vector<8xi64>)
  -> (vector<16xf32>, vector<16xf32>, vector<8xi64>)
{
  // CHECK: avx512.mask.compress {{.*}} : vector<16xf32>
  %0 = avx512.mask.compress %k1, %a1 : vector<16xf32>
  // CHECK: avx512.mask.compress {{.*}} : vector<16xf32>
  %1 = avx512.mask.compress %k1, %a1 {constant_src = dense<5.0> : vector<16xf32>} : vector<16xf32>
  // CHECK: avx512.mask.compress {{.*}} : vector<8xi64>
  %2 = avx512.mask.compress %k2, %a2, %a2 : vector<8xi64>, vector<8xi64>
  return %0, %1, %2 : vector<16xf32>, vector<16xf32>, vector<8xi64>
}
