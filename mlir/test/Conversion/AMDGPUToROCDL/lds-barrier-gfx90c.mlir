// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=triple=amdgpu9.0c-amd-amdhsa | FileCheck %s

// gfx90c sorts after gfx90a, so the version comparison that used to guard the
// inline asm workaround treated it as having the hardware barrier back-off. It
// does not: gfx90c is a Renoir-class APU and lacks FeatureBackOffBarrier, so a
// bare s_barrier lets waits on global memory be introduced around the barrier.

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: llvm.fence syncscope("workgroup") release
  // CHECK-NEXT: llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: ";;;WARNING: BREAKS DEBUG WATCHES\0As_barrier"
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire
  amdgpu.lds_barrier
  func.return
}
