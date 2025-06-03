// REQUIRES: target=x86{{.*}}

// RUN: mlir-opt %s \
// RUN:   -convert-vector-to-llvm="enable-x86vector" -convert-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: llc -mcpu=sapphirerapids | \
// RUN: FileCheck %s

func.func @avx512bf16_dot_128(%src: vector<4xf32>, %a: vector<8xbf16>,
    %b: vector<8xbf16>) -> vector<4xf32> {
  %0 = x86vector.avx512.dot %src, %a, %b : vector<8xbf16> -> vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: avx512bf16_dot_128:
// CHECK: vdpbf16ps{{.*}}%xmm

func.func @avx512bf16_dot_256(%src: vector<8xf32>, %a: vector<16xbf16>,
    %b: vector<16xbf16>) -> vector<8xf32> {
  %0 = x86vector.avx512.dot %src, %a, %b : vector<16xbf16> -> vector<8xf32>
  return %0 : vector<8xf32>
}
// CHECK-LABEL: avx512bf16_dot_256:
// CHECK: vdpbf16ps{{.*}}%ymm

func.func @avx512bf16_dot_512(%src: vector<16xf32>, %a: vector<32xbf16>,
    %b: vector<32xbf16>) -> vector<16xf32> {
  %0 = x86vector.avx512.dot %src, %a, %b : vector<32xbf16> -> vector<16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: avx512bf16_dot_512:
// CHECK: vdpbf16ps{{.*}}%zmm
