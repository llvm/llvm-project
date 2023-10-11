//
// NOTE: this test requires gpu-sm80
//
// with RT lib:
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="enable-runtime-library=true enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format"  \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --e main --entry-point-result=void \
// RUN: | FileCheck %s
//
// without RT lib:
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="enable-runtime-library=false enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format"  \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --e main --entry-point-result=void \
// RUN: | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  // Computes C = A x B with A,B,C sparse CSR.
  func.func @matmulCSR(%A: tensor<8x8xf32, #CSR>,
                       %B: tensor<8x8xf32, #CSR>) -> tensor<8x8xf32, #CSR> {
    %init = tensor.empty() : tensor<8x8xf32, #CSR>
    %C = linalg.matmul
      ins(%A, %B: tensor<8x8xf32, #CSR>,
                  tensor<8x8xf32, #CSR>)
      outs(%init: tensor<8x8xf32, #CSR>) -> tensor<8x8xf32, #CSR>
    return %C: tensor<8x8xf32, #CSR>
  }

  //
  // Main driver.
  //
  func.func @main() {
    llvm.call @mgpuCreateSparseEnv(): () -> ()

    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32

    %t = arith.constant dense<[
       [ 1.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  3.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  5.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  0.0,  6.0,  0.0,  0.0,  0.0],
       [ 0.0,  7.0,  8.0,  0.0,  0.0,  0.0,  0.0,  9.0],
       [ 0.0,  0.0, 10.0,  0.0,  0.0,  0.0, 11.0, 12.0],
       [ 0.0, 13.0, 14.0,  0.0,  0.0,  0.0, 15.0, 16.0]
    ]> : tensor<8x8xf32>
    %Acsr = sparse_tensor.convert %t : tensor<8x8xf32> to tensor<8x8xf32, #CSR>

    %Ccsr = call @matmulCSR(%Acsr, %Acsr) : (tensor<8x8xf32, #CSR>,
                                             tensor<8x8xf32, #CSR>) -> tensor<8x8xf32, #CSR>

    //
    // Verify computed result (expected output, with only 20 nonzeros).
    //
    // CHECK:    ( ( 1, 39, 52, 0, 0, 0, 45, 51 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 16, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 25, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 36, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 117, 158, 0, 0, 0, 135, 144 ),
    // CHECK-SAME: ( 0, 156, 318, 0, 0, 0, 301, 324 ),
    // CHECK-SAME: ( 0, 208, 430, 0, 0, 0, 405, 436 ) )
    // CHECK-NEXT: 20
    %d = sparse_tensor.convert %Ccsr : tensor<8x8xf32, #CSR> to tensor<8x8xf32>
    %v = vector.transfer_read %d[%c0, %c0], %f0: tensor<8x8xf32>, vector<8x8xf32>
    vector.print %v : vector<8x8xf32>
    %nnz = sparse_tensor.number_of_entries %Ccsr : tensor<8x8xf32, #CSR>
    %x = sparse_tensor.number_of_entries %Ccsr : tensor<8x8xf32, #CSR>
    vector.print %nnz : index

    llvm.call @mgpuDestroySparseEnv(): () -> ()
    return
  }
}
