// RUN: mlir-opt %s -convert-gpu-to-agxir -split-input-file | FileCheck %s

// The target's features decide the lowering: on G14 the block barrier is
// native; with +mesa-isa it takes the portable form, a threadgroup barrier.

// CHECK-LABEL: gpu.module @native
// CHECK: agxir.block.barrier
// CHECK-NOT: agxir.threadgroup.barrier
gpu.module @native [#agxir.target<chip = "agx-g14c">] {
  func.func @f() {
    agx.block_barrier
    func.return
  }
}

// -----

// CHECK-LABEL: gpu.module @mesa
// CHECK-NOT: agxir.block.barrier
// CHECK: agxir.threadgroup.barrier
gpu.module @mesa [#agxir.target<chip = "agx-g14c", features = "+mesa-isa">] {
  func.func @f() {
    agx.block_barrier
    func.return
  }
}
