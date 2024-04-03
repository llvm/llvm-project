// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
// with RT lib:
//
// RUN: %{compile} enable-runtime-library=true"  | %{run}
//
// without RT lib:
//
// RUN: %{compile} enable-runtime-library=false" | %{run}

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
    // Verify computed result.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 20
    // CHECK-NEXT: dim = ( 8, 8 )
    // CHECK-NEXT: lvl = ( 8, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 5, 5, 6, 7, 8, 12, 16, 20,
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 6, 7, 2, 3, 4, 1, 2, 6, 7, 1, 2, 6, 7, 1, 2, 6, 7,
    // CHECK-NEXT: values : ( 1, 39, 52, 45, 51, 16, 25, 36, 117, 158, 135, 144, 156, 318, 301, 324, 208, 430, 405, 436,
    // CHECK-NEXT: ----
    sparse_tensor.print %Ccsr : tensor<8x8xf32, #CSR>

    llvm.call @mgpuDestroySparseEnv(): () -> ()
    return
  }
}
