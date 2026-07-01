// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @glb_prefetch0
func.func @glb_prefetch0(%src : memref<64x64xf16, #gpu.address_space<global>>, %i : i64, %j : i64) {
  // CHECK: %[[PTR:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
  // CHECK: rocdl.global.prefetch %[[PTR]], gfx12<lu> : !llvm.ptr<1>
  amdgpu.global_prefetch %src[%i, %j] HT WGP : memref<64x64xf16, #gpu.address_space<global>>
  func.return
}

// CHECK-LABEL: @glb_prefetch1
func.func @glb_prefetch1(%src : memref<64x64xf16, #gpu.address_space<global>>, %i : i64, %j : i64) {
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
  // CHECK: rocdl.global.prefetch %[[PTR]], gfx12<ht|scope_se> : !llvm.ptr<1>
  amdgpu.global_prefetch %src[%i, %j] HT SE speculative : memref<64x64xf16, #gpu.address_space<global>>
  func.return
}

// CHECK-LABEL: @glb_prefetch2
func.func @glb_prefetch2(%src : memref<64x64xf16, #gpu.address_space<global>>, %i : i64, %j : i64) {
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
  // CHECK: rocdl.global.prefetch %{{.*}}, gfx12<scope_dev> : !llvm.ptr<1>
  amdgpu.global_prefetch %src[%i, %j] RT DEV speculative : memref<64x64xf16, #gpu.address_space<global>>
  func.return
}

// CHECK-LABEL: @glb_prefetch3
func.func @glb_prefetch3(%src : memref<64x64xf16, #gpu.address_space<global>>, %i : i64, %j : i64) {
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
  // CHECK: rocdl.global.prefetch %[[PTR]], gfx12<nt_rt|scope_se> : !llvm.ptr<1>
  amdgpu.global_prefetch %src[%i, %j] NT_RT SE speculative : memref<64x64xf16, #gpu.address_space<global>>
  func.return
}
