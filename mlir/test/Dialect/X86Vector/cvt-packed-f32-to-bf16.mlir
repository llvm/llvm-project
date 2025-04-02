// REQUIRES: target=x86{{.*}}

// RUN: mlir-opt %s \
// RUN:   -convert-vector-to-llvm="enable-x86vector" -convert-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: llc -mcpu=sapphirerapids | \
// RUN: FileCheck %s

func.func @avx512bf16_cvt_packed_f32_to_bf16_256(
    %a: vector<8xf32>) -> vector<8xbf16> {
  %0 = x86vector.avx512.cvt.packed.f32_to_bf16 %a : vector<8xf32> -> vector<8xbf16>
  return %0 : vector<8xbf16>
}
// CHECK-LABEL: avx512bf16_cvt_packed_f32_to_bf16_256:
// CHECK: vcvtneps2bf16{{.*}}%xmm

func.func @avx512bf16_cvt_packed_f32_to_bf16_512(
    %a: vector<16xf32>) -> vector<16xbf16> {
  %0 = x86vector.avx512.cvt.packed.f32_to_bf16 %a : vector<16xf32> -> vector<16xbf16>
  return %0 : vector<16xbf16>
}
// CHECK-LABEL: avx512bf16_cvt_packed_f32_to_bf16_512:
// CHECK: vcvtneps2bf16{{.*}}%ymm
