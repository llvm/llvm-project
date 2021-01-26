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

// CHECK-LABEL: define <{ i16, i16 }> @LLVM_x86_vp2intersect_d_512
llvm.func @LLVM_x86_vp2intersect_d_512(%a: vector<16xi32>, %b: vector<16xi32>)
  -> !llvm.struct<packed (i16, i16)>
{
  // CHECK: call { <16 x i1>, <16 x i1> } @llvm.x86.avx512.vp2intersect.d.512(<16 x i32>
  %0 = "llvm_avx512.vp2intersect.d.512"(%a, %b) :
    (vector<16xi32>, vector<16xi32>) -> !llvm.struct<packed (i16, i16)>
  llvm.return %0 : !llvm.struct<packed (i16, i16)>
}

// CHECK-LABEL: define <{ i8, i8 }> @LLVM_x86_vp2intersect_q_512
llvm.func @LLVM_x86_vp2intersect_q_512(%a: vector<8xi64>, %b: vector<8xi64>)
  -> !llvm.struct<packed (i8, i8)>
{
  // CHECK: call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.q.512(<8 x i64>
  %0 = "llvm_avx512.vp2intersect.q.512"(%a, %b) :
    (vector<8xi64>, vector<8xi64>) -> !llvm.struct<packed (i8, i8)>
  llvm.return %0 : !llvm.struct<packed (i8, i8)>
}
