// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx90c | FileCheck %s

// gfx90c needs the inline assembly workaround, just like gfx908.

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: llvm.fence syncscope("workgroup") release
  // CHECK-NEXT: rocdl.s.barrier
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire
  amdgpu.lds_barrier
  func.return
}
