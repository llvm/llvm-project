// RUN: mlir-opt %s -convert-vector-to-llvm="enable-x86vector" | mlir-opt | FileCheck %s

// CHECK-LABEL: func @avx512_mask_rndscale
func.func @avx512_mask_rndscale(
    %src: vector<16xf32>, %a: vector<16xf32>, %b: vector<8xf64>,
    %imm_i16: i16, %imm_i8: i8, %scale_k_i16: i16, %scale_k_i8: i8)
  -> (vector<16xf32>, vector<8xf64>, vector<16xf32>, vector<8xf64>)
{
  %rnd_k = arith.constant 15 : i32
  %rnd = arith.constant 42 : i32
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.rndscale.ps.512"
  %0 = x86vector.avx512.mask.rndscale %src, %rnd_k, %a, %imm_i16, %rnd : vector<16xf32>
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.rndscale.pd.512"
  %1 = x86vector.avx512.mask.rndscale %b, %rnd_k, %b, %imm_i8, %rnd : vector<8xf64>

  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.scalef.ps.512"
  %2 = x86vector.avx512.mask.scalef %a, %a, %a, %scale_k_i16, %rnd : vector<16xf32>
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.scalef.pd.512"
  %3 = x86vector.avx512.mask.scalef %b, %b, %b, %scale_k_i8, %rnd : vector<8xf64>

  // Keep results alive.
  return %0, %1, %2, %3 : vector<16xf32>, vector<8xf64>, vector<16xf32>, vector<8xf64>
}

// CHECK-LABEL: func @avx512_mask_compress
func.func @avx512_mask_compress(
    %k1: vector<16xi1>, %a1: vector<16xf32>, %k2: vector<8xi1>, %a2: vector<8xi64>)
  -> (vector<16xf32>, vector<16xf32>, vector<8xi64>)
{
  // CHECK: llvm.mlir.constant(dense<0.000000e+00> : vector<16xf32>)
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.compress"
  %0 = x86vector.avx512.mask.compress %k1, %a1 : vector<16xf32>
  // CHECK: llvm.mlir.constant(dense<5.000000e+00> : vector<16xf32>)
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.compress"
  %1 = x86vector.avx512.mask.compress %k1, %a1
    {constant_src = dense<5.0> : vector<16xf32>} : vector<16xf32>
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.compress"
  %2 = x86vector.avx512.mask.compress %k2, %a2, %a2 : vector<8xi64>, vector<8xi64>
  return %0, %1, %2 : vector<16xf32>, vector<16xf32>, vector<8xi64>
}

// CHECK-LABEL: func @avx512_vp2intersect
func.func @avx512_vp2intersect(%a: vector<16xi32>, %b: vector<8xi64>)
  -> (vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.vp2intersect.d.512"
  %0, %1 = x86vector.avx512.vp2intersect %a, %a : vector<16xi32>
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512.vp2intersect.q.512"
  %2, %3 = x86vector.avx512.vp2intersect %b, %b : vector<8xi64>
  return %0, %1, %2, %3 : vector<16xi1>, vector<16xi1>, vector<8xi1>, vector<8xi1>
}

// CHECK-LABEL: func @avx512bf16_dot_128
func.func @avx512bf16_dot_128(%src: vector<4xf32>, %a: vector<8xbf16>,
  %b: vector<8xbf16>) -> (vector<4xf32>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512bf16.dpbf16ps.128"
  %0 = x86vector.avx512.dot %src, %a, %b : vector<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avx512bf16_dot_256
func.func @avx512bf16_dot_256(%src: vector<8xf32>, %a: vector<16xbf16>,
  %b: vector<16xbf16>) -> (vector<8xf32>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512bf16.dpbf16ps.256"
  %0 = x86vector.avx512.dot %src, %a, %b : vector<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avx512bf16_dot_512
func.func @avx512bf16_dot_512(%src: vector<16xf32>, %a: vector<32xbf16>,
  %b: vector<32xbf16>) -> (vector<16xf32>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512bf16.dpbf16ps.512"
  %0 = x86vector.avx512.dot %src, %a, %b : vector<32xbf16> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @avx512bf16_cvt_packed_f32_to_bf16_256
func.func @avx512bf16_cvt_packed_f32_to_bf16_256(
  %a: vector<8xf32>) -> (vector<8xbf16>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512bf16.cvtneps2bf16.256"
  %0 = x86vector.avx512.cvt.packed.f32_to_bf16 %a : vector<8xf32> -> vector<8xbf16>
  return %0 : vector<8xbf16>
}

// CHECK-LABEL: func @avx512bf16_cvt_packed_f32_to_bf16_512
func.func @avx512bf16_cvt_packed_f32_to_bf16_512(
  %a: vector<16xf32>) -> (vector<16xbf16>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx512bf16.cvtneps2bf16.512"
  %0 = x86vector.avx512.cvt.packed.f32_to_bf16 %a : vector<16xf32> -> vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @avxbf16_cvt_packed_even_indexed_bf16_to_f32_128
func.func @avxbf16_cvt_packed_even_indexed_bf16_to_f32_128(
  %a: memref<8xbf16>) -> vector<4xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneebf162ps128"
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avxbf16_cvt_packed_even_indexed_bf16_to_f32_256
func.func @avxbf16_cvt_packed_even_indexed_bf16_to_f32_256(
  %a: memref<16xbf16>) -> vector<8xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneebf162ps256"
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avxbf16_cvt_packed_odd_indexed_bf16_to_f32_128
func.func @avxbf16_cvt_packed_odd_indexed_bf16_to_f32_128(
  %a: memref<8xbf16>) -> vector<4xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneobf162ps128"
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avxbf16_cvt_packed_odd_indexed_bf16_to_f32_256
func.func @avxbf16_cvt_packed_odd_indexed_bf16_to_f32_256(
  %a: memref<16xbf16>) -> vector<8xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneobf162ps256"
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avxbf16_bsct_bf16_to_f32_packed_128
func.func @avxbf16_bsct_bf16_to_f32_packed_128(
  %a: memref<1xbf16>) -> vector<4xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vbcstnebf162ps128"
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avxbf16_bsct_bf16_to_f32_packed_256
func.func @avxbf16_bsct_bf16_to_f32_packed_256(
  %a: memref<1xbf16>) -> vector<8xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vbcstnebf162ps256"
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avxf16_cvt_packed_even_indexed_f16_to_f32_128
func.func @avxf16_cvt_packed_even_indexed_f16_to_f32_128(
  %a: memref<8xf16>) -> vector<4xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneeph2ps128"
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<8xf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avxf16_cvt_packed_even_indexed_f16_to_f32_256
func.func @avxf16_cvt_packed_even_indexed_f16_to_f32_256(
  %a: memref<16xf16>) -> vector<8xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneeph2ps256"
  %0 = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avxf16_cvt_packed_odd_indexed_f16_to_f32_128
func.func @avxf16_cvt_packed_odd_indexed_f16_to_f32_128(
  %a: memref<8xf16>) -> vector<4xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneoph2ps128"
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<8xf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avxf16_cvt_packed_odd_indexed_f16_to_f32_256
func.func @avxf16_cvt_packed_odd_indexed_f16_to_f32_256(
  %a: memref<16xf16>) -> vector<8xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vcvtneoph2ps256"
  %0 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avxf16_bsct_f16_to_f32_packed_128
func.func @avxf16_bsct_f16_to_f32_packed_128(
  %a: memref<1xf16>) -> vector<4xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vbcstnesh2ps128"
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @avxf16_bsct_f16_to_f32_packed_256
func.func @avxf16_bsct_f16_to_f32_packed_256(
  %a: memref<1xf16>) -> vector<8xf32>
{
  // CHECK: llvm.call_intrinsic "llvm.x86.vbcstnesh2ps256"
  %0 = x86vector.avx.bcst_to_f32.packed %a : memref<1xf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avx_rsqrt
func.func @avx_rsqrt(%a: vector<8xf32>) -> (vector<8xf32>)
{
  // CHECK: llvm.call_intrinsic "llvm.x86.avx.rsqrt.ps.256"
  %0 = x86vector.avx.rsqrt %a : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avx_dot
func.func @avx_dot(%a: vector<8xf32>, %b: vector<8xf32>) -> (vector<8xf32>)
{
  // CHECK: llvm.mlir.constant(-1 : i8)
  // CHECK: llvm.call_intrinsic "llvm.x86.avx.dp.ps.256"
  %0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @avx_dot_i8_128
func.func @avx_dot_i8_128(%w: vector<4xi32>, %a: vector<16xi8>,
    %b: vector<16xi8>) -> vector<4xi32> {
  // CHECK: llvm.call_intrinsic "llvm.x86.avx2.vpdpbssd.128"
  %0 = x86vector.avx.dot.i8 %w, %a, %b : vector<16xi8> -> vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: func @avx_dot_i8_256
func.func @avx_dot_i8_256(%w: vector<8xi32>, %a: vector<32xi8>,
    %b: vector<32xi8>) -> vector<8xi32> {
  // CHECK: llvm.call_intrinsic "llvm.x86.avx2.vpdpbssd.256"
  %0 = x86vector.avx.dot.i8 %w, %a, %b : vector<32xi8> -> vector<8xi32>
  return %0 : vector<8xi32>
}
