// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | mlir-translate --avx512-mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define <16 x float> @LLVM_x86_avx512_mask_ps_512
llvm.func @LLVM_x86_avx512_mask_ps_512(%a: !llvm.vec<16 x float>,
                                       %b: !llvm.i32,
                                       %c: !llvm.i16)
  -> (!llvm.vec<16 x float>)
{
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>
  %0 = "llvm_avx512.mask.rndscale.ps.512"(%a, %b, %a, %c, %b) :
    (!llvm.vec<16 x float>, !llvm.i32, !llvm.vec<16 x float>, !llvm.i16, !llvm.i32) -> !llvm.vec<16 x float>
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float>
  %1 = "llvm_avx512.mask.scalef.ps.512"(%a, %a, %a, %c, %b) :
    (!llvm.vec<16 x float>, !llvm.vec<16 x float>, !llvm.vec<16 x float>, !llvm.i16, !llvm.i32) -> !llvm.vec<16 x float>
  llvm.return %1: !llvm.vec<16 x float>
}

// CHECK-LABEL: define <8 x double> @LLVM_x86_avx512_mask_pd_512
llvm.func @LLVM_x86_avx512_mask_pd_512(%a: !llvm.vec<8 x double>,
                                       %b: !llvm.i32,
                                       %c: !llvm.i8)
  -> (!llvm.vec<8 x double>)
{
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>
  %0 = "llvm_avx512.mask.rndscale.pd.512"(%a, %b, %a, %c, %b) :
    (!llvm.vec<8 x double>, !llvm.i32, !llvm.vec<8 x double>, !llvm.i8, !llvm.i32) -> !llvm.vec<8 x double>
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double>
  %1 = "llvm_avx512.mask.scalef.pd.512"(%a, %a, %a, %c, %b) :
    (!llvm.vec<8 x double>, !llvm.vec<8 x double>, !llvm.vec<8 x double>, !llvm.i8, !llvm.i32) -> !llvm.vec<8 x double>
  llvm.return %1: !llvm.vec<8 x double>
}
