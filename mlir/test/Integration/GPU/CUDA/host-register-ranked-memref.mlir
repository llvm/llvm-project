// RUN: mlir-opt %s -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void

module attributes {gpu.container_module} {
  func.func @main() {
    %0 = memref.alloc() : memref<64x64xf32>

    // Call host_register with a rank-2 memref.
    gpu.host_register %0 : memref<64x64xf32>

    memref.dealloc %0 : memref<64x64xf32>
    return
  }
}
