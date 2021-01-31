// RUN: mlir-opt %s -convert-vector-to-llvm="enable-avx512" | mlir-opt | FileCheck %s

func @avx512_mask_rndscale(%a: vector<16xf32>, %b: vector<8xf64>, %i32: i32, %i16: i16, %i8: i8)
  -> (vector<16xf32>, vector<8xf64>, vector<16xf32>, vector<8xf64>)
{
  // CHECK: llvm_avx512.mask.rndscale.ps.512
  %0 = avx512.mask.rndscale %a, %i32, %a, %i16, %i32: vector<16xf32>
  // CHECK: llvm_avx512.mask.rndscale.pd.512
  %1 = avx512.mask.rndscale %b, %i32, %b, %i8, %i32: vector<8xf64>

  // CHECK: llvm_avx512.mask.scalef.ps.512
  %2 = avx512.mask.scalef %a, %a, %a, %i16, %i32: vector<16xf32>
  // CHECK: llvm_avx512.mask.scalef.pd.512
  %3 = avx512.mask.scalef %b, %b, %b, %i8, %i32: vector<8xf64>

  // Keep results alive.
  return %0, %1, %2, %3 : vector<16xf32>, vector<8xf64>, vector<16xf32>, vector<8xf64>
}

func @avx512_vp2intersect(%a: vector<16xi32>, %b: vector<8xi64>)
  -> (vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>)
{
  // CHECK: llvm_avx512.vp2intersect.d.512
  %0, %1 = avx512.vp2intersect %a, %a : vector<16xi32>
  // CHECK: llvm_avx512.vp2intersect.q.512
  %2, %3 = avx512.vp2intersect %b, %b : vector<8xi64>
  return %0, %1, %2, %3 : vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>
}
