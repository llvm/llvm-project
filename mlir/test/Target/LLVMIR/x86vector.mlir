// RUN: mlir-opt %s --convert-vector-to-llvm="enable-x86vector" --convert-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | FileCheck %s

// CHECK-LABEL: define <16 x float> @LLVM_x86_avx512_mask_ps_512
func.func @LLVM_x86_avx512_mask_ps_512(
    %src: vector<16xf32>, %a: vector<16xf32>, %b: vector<16xf32>,
    %imm: i16, %scale_k: i16)
  -> (vector<16xf32>)
{
  %rnd_k = arith.constant 15 : i32
  %rnd = arith.constant 42 : i32
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>
  %0 = x86vector.avx512.mask.rndscale %src, %rnd_k, %a, %imm, %rnd : vector<16xf32>
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.scalef.ps.512(<16 x float>
  %1 = x86vector.avx512.mask.scalef %0, %a, %b, %scale_k, %rnd : vector<16xf32>
  return %1 : vector<16xf32>
}

// CHECK-LABEL: define <8 x double> @LLVM_x86_avx512_mask_pd_512
func.func @LLVM_x86_avx512_mask_pd_512(
    %src: vector<8xf64>, %a: vector<8xf64>, %b: vector<8xf64>,
    %imm: i8, %scale_k: i8)
  -> (vector<8xf64>)
{
  %rnd_k = arith.constant 15 : i32
  %rnd = arith.constant 42 : i32
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>
  %0 = x86vector.avx512.mask.rndscale %src, %rnd_k, %a, %imm, %rnd : vector<8xf64>
  // CHECK: call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double>
  %1 = x86vector.avx512.mask.scalef %0, %a, %b, %scale_k, %rnd : vector<8xf64>
  return %1 : vector<8xf64>
}

// CHECK-LABEL: define <16 x float> @LLVM_x86_mask_compress
func.func @LLVM_x86_mask_compress(%k: vector<16xi1>, %a: vector<16xf32>)
  -> vector<16xf32>
{
  // CHECK: call <16 x float> @llvm.x86.avx512.mask.compress.v16f32(
  %0 = x86vector.avx512.mask.compress %k, %a, %a : vector<16xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: define { <16 x i1>, <16 x i1> } @LLVM_x86_vp2intersect_d_512
func.func @LLVM_x86_vp2intersect_d_512(%a: vector<16xi32>, %b: vector<16xi32>)
  -> (vector<16xi1>, vector<16xi1>)
{
  // CHECK: call { <16 x i1>, <16 x i1> } @llvm.x86.avx512.vp2intersect.d.512(<16 x i32>
  %0, %1 = x86vector.avx512.vp2intersect %a, %b : vector<16xi32>
  return %0, %1 : vector<16xi1>, vector<16xi1>
}

// CHECK-LABEL: define { <8 x i1>, <8 x i1> } @LLVM_x86_vp2intersect_q_512
func.func @LLVM_x86_vp2intersect_q_512(%a: vector<8xi64>, %b: vector<8xi64>)
  -> (vector<8 x i1>, vector<8 x i1>)
{
  // CHECK: call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.q.512(<8 x i64>
  %0, %1 = x86vector.avx512.vp2intersect %a, %b : vector<8xi64>
  return %0, %1 : vector<8 x i1>, vector<8 x i1>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avx512bf16_dpbf16ps_128
func.func @LLVM_x86_avx512bf16_dpbf16ps_128(
    %src: vector<4xf32>, %a: vector<8xbf16>, %b: vector<8xbf16>
  ) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(
  %0 = x86vector.avx512.dot %src, %a, %b : vector<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avx512bf16_dpbf16ps_256
func.func @LLVM_x86_avx512bf16_dpbf16ps_256(
    %src: vector<8xf32>, %a: vector<16xbf16>, %b: vector<16xbf16>
  ) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(
  %0 = x86vector.avx512.dot %src, %a, %b : vector<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <16 x float> @LLVM_x86_avx512bf16_dpbf16ps_512
func.func @LLVM_x86_avx512bf16_dpbf16ps_512(
    %src: vector<16xf32>, %a: vector<32xbf16>, %b: vector<32xbf16>
  ) -> vector<16xf32>
{
  // CHECK: call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(
  %0 = x86vector.avx512.dot %src, %a, %b : vector<32xbf16> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: define <8 x bfloat> @LLVM_x86_avx512bf16_cvtneps2bf16_256
func.func @LLVM_x86_avx512bf16_cvtneps2bf16_256(
  %a: vector<8xf32>) -> vector<8xbf16>
{
  // CHECK: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(
  %0 = x86vector.avx512.cvt.packed.f32_to_bf16 %a
    : vector<8xf32> -> vector<8xbf16>
  return %0 : vector<8xbf16>
}

// CHECK-LABEL: define <16 x bfloat> @LLVM_x86_avx512bf16_cvtneps2bf16_512
func.func @LLVM_x86_avx512bf16_cvtneps2bf16_512(
  %a: vector<16xf32>) -> vector<16xbf16>
{
  // CHECK: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(
  %0 = x86vector.avx512.cvt.packed.f32_to_bf16 %a
    : vector<16xf32> -> vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avxbf16_vcvtneebf162ps128
func.func @LLVM_x86_avxbf16_vcvtneebf162ps128(
  %a: memref<8xbf16>) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.vcvtneebf162ps128(
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avxbf16_vcvtneebf162ps256
func.func @LLVM_x86_avxbf16_vcvtneebf162ps256(
  %a: memref<16xbf16>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.vcvtneebf162ps256(
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avxbf16_vcvtneobf162ps128
func.func @LLVM_x86_avxbf16_vcvtneobf162ps128(
  %a: memref<8xbf16>) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.vcvtneobf162ps128(
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avxbf16_vcvtneobf162ps256
func.func @LLVM_x86_avxbf16_vcvtneobf162ps256(
  %a: memref<16xbf16>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.vcvtneobf162ps256(
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avxbf16_vbcstnebf162ps128
func.func @LLVM_x86_avxbf16_vbcstnebf162ps128(
  %a: memref<1xbf16>) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.vbcstnebf162ps128(
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avxbf16_vbcstnebf162ps256
func.func @LLVM_x86_avxbf16_vbcstnebf162ps256(
  %a: memref<1xbf16>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.vbcstnebf162ps256(
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avxf16_vcvtneeph2ps128
func.func @LLVM_x86_avxf16_vcvtneeph2ps128(
  %a: memref<8xf16>) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.vcvtneeph2ps128(
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<8xf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avxf16_vcvtneeph2ps256
func.func @LLVM_x86_avxf16_vcvtneeph2ps256(
  %a: memref<16xf16>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.vcvtneeph2ps256(
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avxf16_vcvtneoph2ps128
func.func @LLVM_x86_avxf16_vcvtneoph2ps128(
  %a: memref<8xf16>) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.vcvtneoph2ps128(
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<8xf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avxf16_vcvtneoph2ps256
func.func @LLVM_x86_avxf16_vcvtneoph2ps256(
  %a: memref<16xf16>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.vcvtneoph2ps256(
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <4 x float> @LLVM_x86_avxf16_vbcstnesh2ps128
func.func @LLVM_x86_avxf16_vbcstnesh2ps128(
  %a: memref<1xf16>) -> vector<4xf32>
{
  // CHECK: call <4 x float> @llvm.x86.vbcstnesh2ps128(
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avxf16_vbcstnesh2ps256
func.func @LLVM_x86_avxf16_vbcstnesh2ps256(
  %a: memref<1xf16>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.vbcstnesh2ps256(
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avx_rsqrt_ps_256
func.func @LLVM_x86_avx_rsqrt_ps_256(%a: vector <8xf32>) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>
  %0 = x86vector.avx.rsqrt %a : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <8 x float> @LLVM_x86_avx_dp_ps_256
func.func @LLVM_x86_avx_dp_ps_256(
    %a: vector<8xf32>, %b: vector<8xf32>
  ) -> vector<8xf32>
{
  // CHECK: call <8 x float> @llvm.x86.avx.dp.ps.256(
  %0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: define <4 x i32> @LLVM_x86_avx2_vpdpbssd_128
func.func @LLVM_x86_avx2_vpdpbssd_128(%w: vector<4xi32>, %a: vector<16xi8>,
    %b: vector<16xi8>) -> vector<4xi32> {
  // CHECK: call <4 x i32> @llvm.x86.avx2.vpdpbssd.128(
  %0 = x86vector.avx.dot.i8 %w, %a, %b : vector<16xi8> -> vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: define <8 x i32> @LLVM_x86_avx2_vpdpbssd_256
func.func @LLVM_x86_avx2_vpdpbssd_256(%w: vector<8xi32>, %a: vector<32xi8>,
    %b: vector<32xi8>) -> vector<8xi32> {
  // CHECK: call <8 x i32> @llvm.x86.avx2.vpdpbssd.256(
  %0 = x86vector.avx.dot.i8 %w, %a, %b : vector<32xi8> -> vector<8xi32>
  return %0 : vector<8xi32>
}
