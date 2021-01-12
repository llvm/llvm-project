// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | mlir-translate --avx512-mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define <16 x float> @LLVM_x86_avx512_mask_ps_512
llvm.func @LLVM_x86_avx512_mask_ps_512(%a: vector<16 x f32>,
                                       %b: i32,
                                       %c: i16)
  -> (vector<16 x f32>)
{
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>
  %0 = "llvm_avx512.mask.rndscale.ps.512"(%a, %b, %a, %c, %b) :
    (vector<16 x f32>, i32, vector<16 x f32>, i16, i32) -> vector<16 x f32>
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float>
  %1 = "llvm_avx512.mask.scalef.ps.512"(%a, %a, %a, %c, %b) :
    (vector<16 x f32>, vector<16 x f32>, vector<16 x f32>, i16, i32) -> vector<16 x f32>
  llvm.return %1: vector<16 x f32>
}

// CHECK-LABEL: define <8 x double> @LLVM_x86_avx512_mask_pd_512
llvm.func @LLVM_x86_avx512_mask_pd_512(%a: vector<8xf64>,
                                       %b: i32,
                                       %c: i8)
  -> (vector<8xf64>)
{
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>
  %0 = "llvm_avx512.mask.rndscale.pd.512"(%a, %b, %a, %c, %b) :
    (vector<8xf64>, i32, vector<8xf64>, i8, i32) -> vector<8xf64>
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double>
  %1 = "llvm_avx512.mask.scalef.pd.512"(%a, %a, %a, %c, %b) :
    (vector<8xf64>, vector<8xf64>, vector<8xf64>, i8, i32) -> vector<8xf64>
  llvm.return %1: vector<8xf64>
}
