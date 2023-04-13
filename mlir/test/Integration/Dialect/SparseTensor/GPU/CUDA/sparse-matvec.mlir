//
// NOTE: this test requires gpu-sm80
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="enable-runtime-library=false parallelization-strategy=dense-outer-loop gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --e main --entry-point-result=void \
// RUN: | FileCheck %s

#CSR = #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>

module {
  // Compute matrix vector y = Ax
  func.func @matvec(%A: tensor<?x?xf64, #CSR>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
    %y_out = linalg.matvec
      ins(%A, %x: tensor<?x?xf64, #CSR>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) -> tensor<?xf64>
    return %y_out : tensor<?xf64>
  }

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

    // Convert to a "sparse" m x n matrix A.
    %A = sparse_tensor.convert %DA : tensor<1024x64xf64> to tensor<?x?xf64, #CSR>

    // Initialize dense vector with n elements:
    //   (1, 2, 3, 4, ..., n)
    %d1 = tensor.dim %A, %c1 : tensor<?x?xf64, #CSR>
    %x = tensor.generate %d1 {
    ^bb0(%i : index):
      %k = arith.addi %i, %c1 : index
      %j = arith.index_cast %k : index to i64
      %f = arith.uitofp %j : i64 to f64
      tensor.yield %f : f64
    } : tensor<?xf64>

    // Initialize dense vector to m zeros.
    %d0 = tensor.dim %A, %c0 : tensor<?x?xf64, #CSR>
    %y = tensor.generate %d0 {
    ^bb0(%i : index):
      tensor.yield %f0 : f64
    } : tensor<?xf64>

    // Call the kernel.
    %0 = call @matvec(%A, %x, %y) : (tensor<?x?xf64, #CSR>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>

    //
    // Sanity check on results.
    //
    // CHECK: ( 87360, 89440, 91520, 93600, 95680, 97760, 99840, 101920, 104000, 106080, 108160, 110240, 112320, 114400, 116480, 118560, 120640, 122720, 124800, 126880, 128960, 131040, 133120, 135200, 137280, 139360, 141440, 143520, 145600, 147680, 149760, 151840, 153920, 156000, 158080, 160160, 162240, 164320, 166400, 168480, 170560, 172640, 174720, 176800, 178880, 180960, 183040, 185120, 187200, 189280, 191360, 193440, 195520, 197600, 199680, 201760, 203840, 205920, 208000, 210080, 212160, 214240, 216320, 218400 )
    //
    %pb0 = vector.transfer_read %0[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb0 : vector<64xf64>

    // Release the resources.
    bufferization.dealloc_tensor %A : tensor<?x?xf64, #CSR>
    return
  }
}
