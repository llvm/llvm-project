//
// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:    --sparse-compiler="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
//
// with RT lib (SoA COO):
//
// RUN:  %{compile} enable-runtime-library=true" | %{run}
// RUN:  %{compile} enable-runtime-library=true gpu-data-transfer-strategy=pinned-dma" | %{run}
// Tracker #64316
// RUNNOT: %{compile} enable-runtime-library=true gpu-data-transfer-strategy=zero-copy" | %{run}
//
// without RT lib (AoS COO): note, may fall back to CPU
//
// RUN: %{compile} enable-runtime-library=false" | %{run}
// RUN: %{compile} enable-runtime-library=false gpu-data-transfer-strategy=pinned-dma" | %{run}
// Tracker #64316
// RUNNOT: %{compile} enable-runtime-library=false gpu-data-transfer-strategy=zero-copy" | %{run}

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
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

  func.func @dump(%mat: tensor<8x8xf32>) {
    %f0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0   : index
    %c1 = arith.constant 1   : index
    %c2 = arith.constant 2   : index
    %c3 = arith.constant 3   : index
    %c4 = arith.constant 4   : index
    %c5 = arith.constant 5   : index
    %c6 = arith.constant 6   : index
    %c7 = arith.constant 7   : index
    %r0 = vector.transfer_read %mat[%c0,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r0 : vector<8xf32>
    %r1 = vector.transfer_read %mat[%c1,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r1 : vector<8xf32>
    %r2 = vector.transfer_read %mat[%c2,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r2 : vector<8xf32>
    %r3 = vector.transfer_read %mat[%c3,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r3 : vector<8xf32>
    %r4 = vector.transfer_read %mat[%c4,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r4 : vector<8xf32>
    %r5 = vector.transfer_read %mat[%c5,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r5 : vector<8xf32>
    %r6 = vector.transfer_read %mat[%c6,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r6 : vector<8xf32>
    %r7 = vector.transfer_read %mat[%c7,%c0], %f0 : tensor<8x8xf32>, vector<8xf32>
    vector.print %r7 : vector<8xf32>
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
    %2 = call @matmulCOO(%Acoo, %DA, %C1) : (tensor<8x8xf32, #SortedCOO>,
                                             tensor<8x8xf32>,
					     tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = call @matmulCSR(%Acsr, %DA, %C1) : (tensor<8x8xf32, #CSR>,
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

    // Release the resources.
    bufferization.dealloc_tensor %Acoo : tensor<8x8xf32, #SortedCOO>
    bufferization.dealloc_tensor %Acsr : tensor<8x8xf32, #CSR>

    llvm.call @mgpuDestroySparseEnv(): () -> ()

    return
  }
}
