// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @glb_prefetch0
func.func @glb_prefetch0(%src : memref<64x64xf16, #gpu.address_space<global>>, %i : i64, %j : i64) {
  // CHECK: rocdl.global.prefetch %{{.*}}, scope 0 : !llvm.ptr<1>
  amdgpu.global_prefetch %src[%i, %j] RT speculative : memref<64x64xf16, #gpu.address_space<global>>
  func.return
}

// CHECK-LABEL: @glb_prefetch1
func.func @glb_prefetch1(%src : memref<64x64xf16, #gpu.address_space<global>>, %i : i64, %j : i64) {
  // CHECK: rocdl.global.prefetch %{{.*}}, scope 3 : !llvm.ptr<1>
  amdgpu.global_prefetch %src[%i, %j] HT : memref<64x64xf16, #gpu.address_space<global>>
  func.return
}
