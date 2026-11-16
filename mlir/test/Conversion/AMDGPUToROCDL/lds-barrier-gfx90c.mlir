// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=arch=amdgpu9.0c-amd-amdhsa | FileCheck %s

// gfx90c needs the inline assembly workaround, just like gfx908.

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: llvm.fence syncscope("workgroup") release
  // CHECK-NEXT: llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: ";;;WARNING: BREAKS DEBUG WATCHES\0As_barrier"
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire
  amdgpu.lds_barrier
  func.return
}
