// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx90c | FileCheck %s

// gfx90c sorts after gfx90a, so the version comparison guarding the inline asm
// workaround treats it as having the hardware barrier back-off. It does not:
// gfx90c is a Renoir-class APU and lacks FeatureBackOffBarrier, so a bare
// s_barrier lets waits on global memory be introduced around the barrier.

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: llvm.fence syncscope("workgroup") release
  // CHECK-NEXT: rocdl.s.barrier
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire
  amdgpu.lds_barrier
  func.return
}
