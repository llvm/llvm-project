// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/170833.
//
// In `gpu-to-llvm`, an `async.yield` operand of type `!gpu.async.token`
// must be lowered to an *event* recorded on the stream that produced it,
// not to the stream pointer itself. Otherwise the host await later calls
// `cuEventSynchronize` on a stream pointer (a no-op that returns an
// error), and the host races against the GPU.
//
// The bug was that two patterns matched `async.yield` with the same
// benefit: the structural rewriter from
// `populateAsyncStructuralTypeConversionsAndLegality` (which only retypes
// operands) and the GPU-aware rewriter (which also creates and records an
// event). When the IR contained `gpu.launch_func` (so other patterns ran
// alongside), the dialect-conversion framework picked the structural one
// for the yield, dropping the event-record on the floor.

module attributes {gpu.container_module} {

  // CHECK-LABEL: llvm.func @yield_launch_token
  // CHECK: %[[stream:.*]] = llvm.call @mgpuStreamCreate
  // CHECK: gpu.launch_func {{.*}} @kmod::@kernel
  // CHECK: %[[event:.*]] = llvm.call @mgpuEventCreate
  // CHECK: llvm.call @mgpuEventRecord(%[[event]], %[[stream]])
  // CHECK: llvm.call @mgpuStreamDestroy(%[[stream]])
  // CHECK: async.yield %[[event]] : !llvm.ptr
  func.func @yield_launch_token(%arg : memref<?xi32>) {
    %c1 = arith.constant 1 : index
    %t, %r = async.execute -> !async.value<!gpu.async.token> {
      %0 = gpu.wait async
      %1 = gpu.launch_func async [%0] @kmod::@kernel
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%arg : memref<?xi32>)
      async.yield %1 : !gpu.async.token
    }
    return
  }

  gpu.module @kmod [#nvvm.target] {
    llvm.func @kernel(%a: !llvm.ptr, %b: !llvm.ptr, %c: i64, %d: i64, %e: i64)
        attributes {gpu.kernel, nvvm.kernel} {
      llvm.return
    }
  }
}
