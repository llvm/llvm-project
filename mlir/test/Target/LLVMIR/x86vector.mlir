// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define <16 x float> @LLVM_x86_avx512_mask_ps_512
llvm.func @LLVM_x86_avx512_mask_ps_512(%a: vector<16 x f32>,
                                       %c: i16)
  -> (vector<16 x f32>)
{
  %b = llvm.mlir.constant(42 : i32) : i32
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>
  %0 = "x86vector.avx512.intr.mask.rndscale.ps.512"(%a, %b, %a, %c, %b) :
    (vector<16 x f32>, i32, vector<16 x f32>, i16, i32) -> vector<16 x f32>
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float>
  %1 = "x86vector.avx512.intr.mask.scalef.ps.512"(%a, %a, %a, %c, %b) :
    (vector<16 x f32>, vector<16 x f32>, vector<16 x f32>, i16, i32) -> vector<16 x f32>
  llvm.return %1: vector<16 x f32>
}

// CHECK-LABEL: define <8 x double> @LLVM_x86_avx512_mask_pd_512
llvm.func @LLVM_x86_avx512_mask_pd_512(%a: vector<8xf64>,
                                       %c: i8)
  -> (vector<8xf64>)
{
  %b = llvm.mlir.constant(42 : i32) : i32
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>
  %0 = "x86vector.avx512.intr.mask.rndscale.pd.512"(%a, %b, %a, %c, %b) :
    (vector<8xf64>, i32, vector<8xf64>, i8, i32) -> vector<8xf64>
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double>
  %1 = "x86vector.avx512.intr.mask.scalef.pd.512"(%a, %a, %a, %c, %b) :
    (vector<8xf64>, vector<8xf64>, vector<8xf64>, i8, i32) -> vector<8xf64>
  llvm.return %1: vector<8xf64>
}

// CHECK-LABEL: define <16 x float> @LLVM_x86_mask_compress
llvm.func @LLVM_x86_mask_compress(%k: vector<16xi1>, %a: vector<16xf32>)
  -> vector<16xf32>
{
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.compress.v16f32(
  %0 = "x86vector.avx512.intr.mask.compress"(%a, %a, %k) :
    (vector<16xf32>, vector<16xf32>, vector<16xi1>) -> vector<16xf32>
  llvm.return %0 : vector<16xf32>
}

// CHECK-LABEL: define { <16 x i1>, <16 x i1> } @LLVM_x86_vp2intersect_d_512
llvm.func @LLVM_x86_vp2intersect_d_512(%a: vector<16xi32>, %b: vector<16xi32>)
  -> !llvm.struct<(vector<16 x i1>, vector<16 x i1>)>
{
  // CHECK: call { <16 x i1>, <16 x i1> } @llvm.x86.avx512.vp2intersect.d.512(<16 x i32>
  %0 = "x86vector.avx512.intr.vp2intersect.d.512"(%a, %b) :
    (vector<16xi32>, vector<16xi32>) -> !llvm.struct<(vector<16 x i1>, vector<16 x i1>)>
  llvm.return %0 : !llvm.struct<(vector<16 x i1>, vector<16 x i1>)>
}

// CHECK-LABEL: define { <8 x i1>, <8 x i1> } @LLVM_x86_vp2intersect_q_512
llvm.func @LLVM_x86_vp2intersect_q_512(%a: vector<8xi64>, %b: vector<8xi64>)
  -> !llvm.struct<(vector<8 x i1>, vector<8 x i1>)>
{
  // CHECK: call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.q.512(<8 x i64>
  %0 = "x86vector.avx512.intr.vp2intersect.q.512"(%a, %b) :
    (vector<8xi64>, vector<8xi64>) -> !llvm.struct<(vector<8 x i1>, vector<8 x i1>)>
  llvm.return %0 : !llvm.struct<(vector<8 x i1>, vector<8 x i1>)>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avx512bf16_dpbf16ps_128
llvm.func @LLVM_x86_avx512bf16_dpbf16ps_128(
    %arg0: vector<4xf32>, %arg1: vector<8xbf16>, %arg2: vector<8xbf16>
  ) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(
  %0 = "x86vector.avx512.intr.dpbf16ps.128"(%arg0, %arg1, %arg2)
    : (vector<4xf32>, vector<8xbf16>, vector<8xbf16>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avx512bf16_dpbf16ps_256
llvm.func @LLVM_x86_avx512bf16_dpbf16ps_256(
    %arg0: vector<8xf32>, %arg1: vector<16xbf16>, %arg2: vector<16xbf16>
  ) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(
  %0 = "x86vector.avx512.intr.dpbf16ps.256"(%arg0, %arg1, %arg2)
    : (vector<8xf32>, vector<16xbf16>, vector<16xbf16>) -> vector<8xf32>
  llvm.return %0 : vector<8xf32>
}

// CHECK-LABEL: define <16 x float> @LLVM_x86_avx512bf16_dpbf16ps_512
llvm.func @LLVM_x86_avx512bf16_dpbf16ps_512(
    %arg0: vector<16xf32>, %arg1: vector<32xbf16>, %arg2: vector<32xbf16>
  ) -> vector<16xf32>
{
  // CHECK: call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(
  %0 = "x86vector.avx512.intr.dpbf16ps.512"(%arg0, %arg1, %arg2)
    : (vector<16xf32>, vector<32xbf16>, vector<32xbf16>) -> vector<16xf32>
  llvm.return %0 : vector<16xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avx_rsqrt_ps_256
llvm.func @LLVM_x86_avx_rsqrt_ps_256(%a: vector <8xf32>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>
  %0 = "x86vector.avx.intr.rsqrt.ps.256"(%a) : (vector<8xf32>) -> (vector<8xf32>)
  llvm.return %0 : vector<8xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avx_dp_ps_256
llvm.func @LLVM_x86_avx_dp_ps_256(
    %arg0: vector<8xf32>, %arg1: vector<8xf32>
  ) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.avx.dp.ps.256(
  %0 = llvm.mlir.constant(-1 : i8) : i8
  %1 = "x86vector.avx.intr.dp.ps.256"(%arg0, %arg1, %0) : (vector<8xf32>, vector<8xf32>, i8) -> vector<8xf32>
  llvm.return %1 : vector<8xf32>
}
