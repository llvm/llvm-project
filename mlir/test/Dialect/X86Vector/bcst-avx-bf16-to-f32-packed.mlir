// REQUIRES: target=x86{{.*}}

// RUN: mlir-opt %s \
// RUN:   -convert-vector-to-llvm="enable-x86vector" -convert-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: llc -mcpu=sierraforest | \
// RUN: FileCheck %s

func.func @avxbf16_bcst_bf16_to_f32_packed_128(%arg0: !llvm.ptr) -> vector<4xf32> {
  %0 = x86vector.avx.bcst.bf16_to_f32.packed %arg0 : !llvm.ptr -> vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: avxbf16_bcst_bf16_to_f32_packed_128:
// CHECK: vbcstnebf162ps{{.*}}%xmm

func.func @avxbf16_bcst_bf16_to_f32_packed_256(%arg0: !llvm.ptr) -> vector<8xf32> {
  %0 = x86vector.avx.bcst.bf16_to_f32.packed %arg0 : !llvm.ptr -> vector<8xf32>
  return %0 : vector<8xf32>
}
// CHECK-LABEL: avxbf16_bcst_bf16_to_f32_packed_256:
// CHECK: vbcstnebf162ps{{.*}}%ymm
