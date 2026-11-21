// RUN: mlir-opt %s -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format allow-pattern-rollback=0" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// A rank-zero memref registers one element, even though its descriptor has no
// size or stride entries.
func.func @main() {
  %buffer = memref.alloc() : memref<f32>
  %unranked = memref.cast %buffer : memref<f32> to memref<*xf32>
  gpu.host_register %unranked : memref<*xf32>
  %registered = call @mgpuMemHostIsRegistered(%buffer) : (memref<f32>) -> i32
  // CHECK: {{^1$}}
  vector.print %registered : i32

  gpu.host_unregister %unranked : memref<*xf32>
  %unregistered = call @mgpuMemHostIsRegistered(%buffer) : (memref<f32>) -> i32
  // CHECK-NEXT: {{^0$}}
  vector.print %unregistered : i32

  memref.dealloc %buffer : memref<f32>
  return
}

func.func private @mgpuMemHostIsRegistered(memref<f32>) -> i32
    attributes {llvm.bareptr}
