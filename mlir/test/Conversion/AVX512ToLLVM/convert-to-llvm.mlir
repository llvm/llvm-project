// RUN: mlir-opt %s -convert-avx512-to-llvm | mlir-opt | FileCheck %s

func @avx512_mask_rndscale(%a: vector<16xf32>, %b: vector<8xf64>, %i32: i32, %i16: i16, %i8: i8)
  -> (vector<16xf32>, vector<8xf64>)
{
  // CHECK: llvm_avx512.mask.rndscale.ps.512
  %0 = avx512.mask.rndscale %a, %i32, %a, %i16, %i32: vector<16xf32>
  // CHECK: llvm_avx512.mask.rndscale.pd.512
  %1 = avx512.mask.rndscale %b, %i32, %b, %i8, %i32: vector<8xf64>

  // CHECK: llvm_avx512.mask.scalef.ps.512
  %a0 = avx512.mask.scalef %a, %a, %a, %i16, %i32: vector<16xf32>
  // CHECK: llvm_avx512.mask.scalef.pd.512
  %a1 = avx512.mask.scalef %b, %b, %b, %i8, %i32: vector<8xf64>

  return %a0, %a1: vector<16xf32>, vector<8xf64>
}
