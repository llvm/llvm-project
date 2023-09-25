//
// NOTE: this test requires gpu-sm80
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="enable-runtime-library=false parallelization-strategy=dense-outer-loop gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --e main --entry-point-result=void \
// RUN: | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

module {
  // Compute matrix vector y = Ax
  func.func @matvec(%A: tensor<1024x64xf64, #CSR>, %x: tensor<64xf64>, %y_in: tensor<1024xf64>) -> tensor<1024xf64> {
    %y_out = linalg.matvec
      ins(%A, %x: tensor<1024x64xf64, #CSR>, tensor<64xf64>)
      outs(%y_in: tensor<1024xf64>) -> tensor<1024xf64>
    return %y_out : tensor<1024xf64>
  }

  memref.global "private" constant @__constant_64xf64 : memref<64xf64> = dense<1.000000e+00> {alignment = 64 : i64}

  func.func @main() {
    %f0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Stress test with a dense matrix DA.
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f64
      tensor.yield %f : f64
    } : tensor<1024x64xf64>

    // Convert to a "sparse" 1024 x 64 matrix A.
    %A = sparse_tensor.convert %DA : tensor<1024x64xf64> to tensor<1024x64xf64, #CSR>

    // Initialize dense vector to 1024 zeros.
    %y = tensor.generate {
    ^bb0(%i : index):
      tensor.yield %f0 : f64
    } : tensor<1024xf64>

    // Call the kernel with an vector taken from global memory.
    %xbuf = memref.get_global @__constant_64xf64 : memref<64xf64>
    %x = bufferization.to_tensor %xbuf restrict : memref<64xf64>
    %0 = call @matvec(%A, %x, %y) : (tensor<1024x64xf64, #CSR>, tensor<64xf64>, tensor<1024xf64>) -> tensor<1024xf64>

    //
    // Sanity check on results.
    //
    // CHECK: ( 2016, 2080, 2144, 2208, 2272, 2336, 2400, 2464, 2528, 2592, 2656, 2720, 2784, 2848, 2912, 2976, 3040, 3104, 3168, 3232, 3296, 3360, 3424, 3488, 3552, 3616, 3680, 3744, 3808, 3872, 3936, 4000, 4064, 4128, 4192, 4256, 4320, 4384, 4448, 4512, 4576, 4640, 4704, 4768, 4832, 4896, 4960, 5024, 5088, 5152, 5216, 5280, 5344, 5408, 5472, 5536, 5600, 5664, 5728, 5792, 5856, 5920, 5984, 6048 )
    //
    %pb0 = vector.transfer_read %0[%c0], %f0 : tensor<1024xf64>, vector<64xf64>
    vector.print %pb0 : vector<64xf64>

    // Release the resources.
    bufferization.dealloc_tensor %A : tensor<1024x64xf64, #CSR>
    return
  }
}
