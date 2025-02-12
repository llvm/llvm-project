// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
// with RT lib (SoA COO):
//
// RUN: %{compile} enable-runtime-library=true"  | %{run}
//
// without RT lib (AoS COO): note, may fall back to CPU
//
// RUN: %{compile} enable-runtime-library=false" | %{run}

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  // Computes C = A x B with A sparse COO.
  func.func @matmulCOO(%A: tensor<8x8xf32, #SortedCOO>,
                       %B: tensor<8x8xf32>,
                       %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul
      ins(%A, %B: tensor<8x8xf32, #SortedCOO>, tensor<8x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D: tensor<8x8xf32>
  }

  // Computes C = A x B with A sparse CSR.
  func.func @matmulCSR(%A: tensor<8x8xf32, #CSR>,
                       %B: tensor<8x8xf32>,
                       %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul
      ins(%A, %B: tensor<8x8xf32, #CSR>, tensor<8x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D: tensor<8x8xf32>
  }

  // Computes C = A x B with A sparse CSC.
  func.func @matmulCSC(%A: tensor<8x8xf32, #CSC>,
                       %B: tensor<8x8xf32>,
                       %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul
      ins(%A, %B: tensor<8x8xf32, #CSC>, tensor<8x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D: tensor<8x8xf32>
  }

  // Helper to dump dense tensor as series of vectors.
  func.func @dump(%mat: tensor<8x8xf32>) {
    %f0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0   : index
    %c1 = arith.constant 1   : index
    %c8 = arith.constant 8   : index
    scf.for %i = %c0 to %c8 step %c1 {
      %v = vector.transfer_read %mat[%i,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
      vector.print %v : vector<8xf32>
    }
    return
  }

  //
  // Main driver.
  //
  func.func @main() {
    llvm.call @mgpuCreateSparseEnv(): () -> ()
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32

    // Stress test with a dense matrix DA.
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f32
      tensor.yield %f : f32
    } : tensor<8x8xf32>

    // Convert to a "sparse" matrix A.
    %Acoo = sparse_tensor.convert %DA : tensor<8x8xf32> to tensor<8x8xf32, #SortedCOO>
    %Acsr = sparse_tensor.convert %DA : tensor<8x8xf32> to tensor<8x8xf32, #CSR>
    %Acsc = sparse_tensor.convert %DA : tensor<8x8xf32> to tensor<8x8xf32, #CSC>

    // Initial C matrices.
    %C0 = tensor.generate {
    ^bb0(%i: index, %j: index):
      tensor.yield %f0 : f32
    } : tensor<8x8xf32>
    %C1 = tensor.generate {
    ^bb0(%i: index, %j: index):
      tensor.yield %f1 : f32
    } : tensor<8x8xf32>

     // Call the kernels.
    %0 = call @matmulCOO(%Acoo, %DA, %C0) : (tensor<8x8xf32, #SortedCOO>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>
    %1 = call @matmulCSR(%Acsr, %DA, %C0) : (tensor<8x8xf32, #CSR>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>
    %2 = call @matmulCSC(%Acsc, %DA, %C0) : (tensor<8x8xf32, #CSC>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = call @matmulCOO(%Acoo, %DA, %C1) : (tensor<8x8xf32, #SortedCOO>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>
    %4 = call @matmulCSR(%Acsr, %DA, %C1) : (tensor<8x8xf32, #CSR>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>
    %5 = call @matmulCSC(%Acsc, %DA, %C1) : (tensor<8x8xf32, #CSC>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>

    //
    // Sanity check on results.
    //
    // CHECK:      ( 140, 168, 196, 224, 252, 280, 308, 336 )
    // CHECK-NEXT: ( 168, 204, 240, 276, 312, 348, 384, 420 )
    // CHECK-NEXT: ( 196, 240, 284, 328, 372, 416, 460, 504 )
    // CHECK-NEXT: ( 224, 276, 328, 380, 432, 484, 536, 588 )
    // CHECK-NEXT: ( 252, 312, 372, 432, 492, 552, 612, 672 )
    // CHECK-NEXT: ( 280, 348, 416, 484, 552, 620, 688, 756 )
    // CHECK-NEXT: ( 308, 384, 460, 536, 612, 688, 764, 840 )
    // CHECK-NEXT: ( 336, 420, 504, 588, 672, 756, 840, 924 )
    //
    // CHECK:      ( 140, 168, 196, 224, 252, 280, 308, 336 )
    // CHECK-NEXT: ( 168, 204, 240, 276, 312, 348, 384, 420 )
    // CHECK-NEXT: ( 196, 240, 284, 328, 372, 416, 460, 504 )
    // CHECK-NEXT: ( 224, 276, 328, 380, 432, 484, 536, 588 )
    // CHECK-NEXT: ( 252, 312, 372, 432, 492, 552, 612, 672 )
    // CHECK-NEXT: ( 280, 348, 416, 484, 552, 620, 688, 756 )
    // CHECK-NEXT: ( 308, 384, 460, 536, 612, 688, 764, 840 )
    // CHECK-NEXT: ( 336, 420, 504, 588, 672, 756, 840, 924 )
    //
    // CHECK:      ( 140, 168, 196, 224, 252, 280, 308, 336 )
    // CHECK-NEXT: ( 168, 204, 240, 276, 312, 348, 384, 420 )
    // CHECK-NEXT: ( 196, 240, 284, 328, 372, 416, 460, 504 )
    // CHECK-NEXT: ( 224, 276, 328, 380, 432, 484, 536, 588 )
    // CHECK-NEXT: ( 252, 312, 372, 432, 492, 552, 612, 672 )
    // CHECK-NEXT: ( 280, 348, 416, 484, 552, 620, 688, 756 )
    // CHECK-NEXT: ( 308, 384, 460, 536, 612, 688, 764, 840 )
    // CHECK-NEXT: ( 336, 420, 504, 588, 672, 756, 840, 924 )
    //
    // CHECK:      ( 141, 169, 197, 225, 253, 281, 309, 337 )
    // CHECK-NEXT: ( 169, 205, 241, 277, 313, 349, 385, 421 )
    // CHECK-NEXT: ( 197, 241, 285, 329, 373, 417, 461, 505 )
    // CHECK-NEXT: ( 225, 277, 329, 381, 433, 485, 537, 589 )
    // CHECK-NEXT: ( 253, 313, 373, 433, 493, 553, 613, 673 )
    // CHECK-NEXT: ( 281, 349, 417, 485, 553, 621, 689, 757 )
    // CHECK-NEXT: ( 309, 385, 461, 537, 613, 689, 765, 841 )
    // CHECK-NEXT: ( 337, 421, 505, 589, 673, 757, 841, 925 )
    //
    // CHECK:      ( 141, 169, 197, 225, 253, 281, 309, 337 )
    // CHECK-NEXT: ( 169, 205, 241, 277, 313, 349, 385, 421 )
    // CHECK-NEXT: ( 197, 241, 285, 329, 373, 417, 461, 505 )
    // CHECK-NEXT: ( 225, 277, 329, 381, 433, 485, 537, 589 )
    // CHECK-NEXT: ( 253, 313, 373, 433, 493, 553, 613, 673 )
    // CHECK-NEXT: ( 281, 349, 417, 485, 553, 621, 689, 757 )
    // CHECK-NEXT: ( 309, 385, 461, 537, 613, 689, 765, 841 )
    // CHECK-NEXT: ( 337, 421, 505, 589, 673, 757, 841, 925 )
    //
    // CHECK:      ( 141, 169, 197, 225, 253, 281, 309, 337 )
    // CHECK-NEXT: ( 169, 205, 241, 277, 313, 349, 385, 421 )
    // CHECK-NEXT: ( 197, 241, 285, 329, 373, 417, 461, 505 )
    // CHECK-NEXT: ( 225, 277, 329, 381, 433, 485, 537, 589 )
    // CHECK-NEXT: ( 253, 313, 373, 433, 493, 553, 613, 673 )
    // CHECK-NEXT: ( 281, 349, 417, 485, 553, 621, 689, 757 )
    // CHECK-NEXT: ( 309, 385, 461, 537, 613, 689, 765, 841 )
    // CHECK-NEXT: ( 337, 421, 505, 589, 673, 757, 841, 925 )
    //
    call @dump(%0) : (tensor<8x8xf32>) -> ()
    call @dump(%1) : (tensor<8x8xf32>) -> ()
    call @dump(%2) : (tensor<8x8xf32>) -> ()
    call @dump(%3) : (tensor<8x8xf32>) -> ()
    call @dump(%4) : (tensor<8x8xf32>) -> ()
    call @dump(%5) : (tensor<8x8xf32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %Acoo : tensor<8x8xf32, #SortedCOO>
    bufferization.dealloc_tensor %Acsr : tensor<8x8xf32, #CSR>
    bufferization.dealloc_tensor %Acsc : tensor<8x8xf32, #CSC>

    llvm.call @mgpuDestroySparseEnv(): () -> ()

    return
  }
}
