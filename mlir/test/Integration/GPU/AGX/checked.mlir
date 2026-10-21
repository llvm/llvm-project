// RUN: mlir-opt %s -gpu-kernel-outlining \
// RUN:   -agxir-attach-target="chip=%chip features=+agx-checked" \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(convert-gpu-to-agxir,reconcile-unrealized-casts),ensure-debug-info-scope-on-llvm-func,gpu-to-llvm,reconcile-unrealized-casts,gpu-module-to-binary)' \
// RUN: | env ZLUDA_AGX_DUMP= mlir-runner --shared-libs=%mlir_agx_runtime \
// RUN:   --shared-libs=%mlir_runner_utils --entry-point-result=void 2>&1 \
// RUN: | FileCheck %s

// Checked mode through MLIR's GPU runtime wrappers on an Apple GPU: the
// launch goes through mgpuLaunchKernel to ZLUDA's cuLaunchKernel, which fills
// the kernel's hidden arguments, and the bad access the kernel reports is
// printed at mgpuStreamSynchronize with the line of the memref.store here.

// CHECK: zluda_agx: out of bounds store of 4 bytes at {{.*}} by thread (16, 0, 0) of block (0, 0, 0)
// CHECK-NEXT: at main_kernel {{.*}}checked.mlir:[[# @LINE + 10]]:7

module attributes {gpu.container_module} {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %buf = gpu.alloc () : memref<16xi32>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
      %v = arith.index_cast %tx : index to i32
      memref.store %v, %buf[%tx] : memref<16xi32>
      gpu.terminator
    }
    gpu.dealloc %buf : memref<16xi32>
    return
  }
}
