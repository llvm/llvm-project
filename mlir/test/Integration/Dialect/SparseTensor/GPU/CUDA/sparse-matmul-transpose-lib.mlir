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

#transpose_a = [
  affine_map<(i, j, k) -> (k, i)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]

#transpose_b = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (j, k)>,
  affine_map<(i, j, k) -> (i, j)>
]

#transpose_ab = [
  affine_map<(i, j, k) -> (k, i)>,
  affine_map<(i, j, k) -> (j, k)>,
  affine_map<(i, j, k) -> (i, j)>
]

module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  // Compute C = A^T B with A in COO format.
  func.func @matmulCOO_transpose_a(%A: tensor<4x8xf32, #SortedCOO>,
                                   %B: tensor<4x8xf32>,
                                   %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_a
      ins(%A, %B: tensor<4x8xf32, #SortedCOO>, tensor<4x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A^T B with A in CSR format.
  func.func @matmulCSR_transpose_a(%A: tensor<4x8xf32, #CSR>,
                                   %B: tensor<4x8xf32>,
                                   %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_a
      ins(%A, %B: tensor<4x8xf32, #CSR>, tensor<4x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A^T B with A in CSC format.
  func.func @matmulCSC_transpose_a(%A: tensor<4x8xf32, #CSC>,
                                   %B: tensor<4x8xf32>,
                                   %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_a
      ins(%A, %B: tensor<4x8xf32, #CSC>, tensor<4x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A B^T with A in COO format.
  func.func @matmulCOO_transpose_b(%A: tensor<8x4xf32, #SortedCOO>,
                                   %B: tensor<8x4xf32>,
                                   %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_b
      ins(%A, %B: tensor<8x4xf32, #SortedCOO>, tensor<8x4xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A B^T with A in CSR format.
  func.func @matmulCSR_transpose_b(%A: tensor<8x4xf32, #CSR>,
                                   %B: tensor<8x4xf32>,
                                   %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_b
      ins(%A, %B: tensor<8x4xf32, #CSR>, tensor<8x4xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A B^T with A in CSC format.
  func.func @matmulCSC_transpose_b(%A: tensor<8x4xf32, #CSC>,
                                   %B: tensor<8x4xf32>,
                                   %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_b
      ins(%A, %B: tensor<8x4xf32, #CSC>, tensor<8x4xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A^T B^T with A in COO format.
  func.func @matmulCOO_transpose_ab(%A: tensor<4x8xf32, #SortedCOO>,
                                    %B: tensor<8x4xf32>,
                                    %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_ab
      ins(%A, %B: tensor<4x8xf32, #SortedCOO>, tensor<8x4xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A^T B^T with A in CSR format.
  func.func @matmulCSR_transpose_ab(%A: tensor<4x8xf32, #CSR>,
                                    %B: tensor<8x4xf32>,
                                    %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_ab
      ins(%A, %B: tensor<4x8xf32, #CSR>, tensor<8x4xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
  }

  // Compute C = A^T B^T with A in CSC format.
  func.func @matmulCSC_transpose_ab(%A: tensor<4x8xf32, #CSC>,
                                    %B: tensor<8x4xf32>,
                                    %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul indexing_maps = #transpose_ab
      ins(%A, %B: tensor<4x8xf32, #CSC>, tensor<8x4xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D : tensor<8x8xf32>
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
    llvm.call @mgpuCreateSparseEnv() : () -> ()
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32

    // Stress test with rectangular dense matrices DA and DAT.
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f32
      tensor.yield %f : f32
    } : tensor<8x4xf32>
    %DAT = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f32
      tensor.yield %f : f32
    } : tensor<4x8xf32>

    // Convert to "sparse" matrices A and AT.
    %Acoo = sparse_tensor.convert %DA : tensor<8x4xf32> to tensor<8x4xf32, #SortedCOO>
    %Acsr = sparse_tensor.convert %DA : tensor<8x4xf32> to tensor<8x4xf32, #CSR>
    %Acsc = sparse_tensor.convert %DA : tensor<8x4xf32> to tensor<8x4xf32, #CSC>
    %ATcoo = sparse_tensor.convert %DAT : tensor<4x8xf32> to tensor<4x8xf32, #SortedCOO>
    %ATcsr = sparse_tensor.convert %DAT : tensor<4x8xf32> to tensor<4x8xf32, #CSR>
    %ATcsc = sparse_tensor.convert %DAT : tensor<4x8xf32> to tensor<4x8xf32, #CSC>

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
    %0 = call @matmulCOO_transpose_a(%ATcoo, %DAT, %C0) : (tensor<4x8xf32, #SortedCOO>,
                                                           tensor<4x8xf32>,
                                                           tensor<8x8xf32>) -> tensor<8x8xf32>
    %1 = call @matmulCSR_transpose_a(%ATcsr, %DAT, %C0) : (tensor<4x8xf32, #CSR>,
                                                           tensor<4x8xf32>,
                                                           tensor<8x8xf32>) -> tensor<8x8xf32>
    %2 = call @matmulCSC_transpose_a(%ATcsc, %DAT, %C0) : (tensor<4x8xf32, #CSC>,
                                                           tensor<4x8xf32>,
                                                           tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = call @matmulCOO_transpose_a(%ATcoo, %DAT, %C1) : (tensor<4x8xf32, #SortedCOO>,
                                                           tensor<4x8xf32>,
                                                           tensor<8x8xf32>) -> tensor<8x8xf32>
    %4 = call @matmulCSR_transpose_a(%ATcsr, %DAT, %C1) : (tensor<4x8xf32, #CSR>,
                                                           tensor<4x8xf32>,
                                                           tensor<8x8xf32>) -> tensor<8x8xf32>
    %5 = call @matmulCSC_transpose_a(%ATcsc, %DAT, %C1) : (tensor<4x8xf32, #CSC>,
                                                           tensor<4x8xf32>,
                                                           tensor<8x8xf32>) -> tensor<8x8xf32>
    %6 = call @matmulCOO_transpose_b(%Acoo, %DA, %C0) : (tensor<8x4xf32, #SortedCOO>,
                                                         tensor<8x4xf32>,
                                                         tensor<8x8xf32>) -> tensor<8x8xf32>
    %7 = call @matmulCSR_transpose_b(%Acsr, %DA, %C0) : (tensor<8x4xf32, #CSR>,
                                                         tensor<8x4xf32>,
                                                         tensor<8x8xf32>) -> tensor<8x8xf32>
    %8 = call @matmulCSC_transpose_b(%Acsc, %DA, %C0) : (tensor<8x4xf32, #CSC>,
                                                         tensor<8x4xf32>,
                                                         tensor<8x8xf32>) -> tensor<8x8xf32>
    %9 = call @matmulCOO_transpose_b(%Acoo, %DA, %C1) : (tensor<8x4xf32, #SortedCOO>,
                                                         tensor<8x4xf32>,
                                                         tensor<8x8xf32>) -> tensor<8x8xf32>
    %10 = call @matmulCSR_transpose_b(%Acsr, %DA, %C1) : (tensor<8x4xf32, #CSR>,
                                                          tensor<8x4xf32>,
                                                          tensor<8x8xf32>) -> tensor<8x8xf32>
    %11 = call @matmulCSC_transpose_b(%Acsc, %DA, %C1) : (tensor<8x4xf32, #CSC>,
                                                          tensor<8x4xf32>,
                                                          tensor<8x8xf32>) -> tensor<8x8xf32>
    %12 = call @matmulCOO_transpose_ab(%ATcoo, %DA, %C0) : (tensor<4x8xf32, #SortedCOO>,
                                                            tensor<8x4xf32>,
                                                            tensor<8x8xf32>) -> tensor<8x8xf32>
    %13 = call @matmulCSR_transpose_ab(%ATcsr, %DA, %C0) : (tensor<4x8xf32, #CSR>,
                                                            tensor<8x4xf32>,
                                                            tensor<8x8xf32>) -> tensor<8x8xf32>
    %14 = call @matmulCSC_transpose_ab(%ATcsc, %DA, %C0) : (tensor<4x8xf32, #CSC>,
                                                            tensor<8x4xf32>,
                                                            tensor<8x8xf32>) -> tensor<8x8xf32>
    %15 = call @matmulCOO_transpose_ab(%ATcoo, %DA, %C1) : (tensor<4x8xf32, #SortedCOO>,
                                                            tensor<8x4xf32>,
                                                            tensor<8x8xf32>) -> tensor<8x8xf32>
    %16 = call @matmulCSR_transpose_ab(%ATcsr, %DA, %C1) : (tensor<4x8xf32, #CSR>,
                                                            tensor<8x4xf32>,
                                                            tensor<8x8xf32>) -> tensor<8x8xf32>
    %17 = call @matmulCSC_transpose_ab(%ATcsc, %DA, %C1) : (tensor<4x8xf32, #CSC>,
                                                            tensor<8x4xf32>,
                                                            tensor<8x8xf32>) -> tensor<8x8xf32>

    //
    // Sanity check on results.
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // A B^T with C initialized to zeros, then ones.
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // A^T B^T with C initialized to zeros, then ones.
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 14, 20, 26, 32, 38, 44, 50, 56 )
    // CHECK-NEXT: ( 20, 30, 40, 50, 60, 70, 80, 90 )
    // CHECK-NEXT: ( 26, 40, 54, 68, 82, 96, 110, 124 )
    // CHECK-NEXT: ( 32, 50, 68, 86, 104, 122, 140, 158 )
    // CHECK-NEXT: ( 38, 60, 82, 104, 126, 148, 170, 192 )
    // CHECK-NEXT: ( 44, 70, 96, 122, 148, 174, 200, 226 )
    // CHECK-NEXT: ( 50, 80, 110, 140, 170, 200, 230, 260 )
    // CHECK-NEXT: ( 56, 90, 124, 158, 192, 226, 260, 294 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    // CHECK:      ( 15, 21, 27, 33, 39, 45, 51, 57 )
    // CHECK-NEXT: ( 21, 31, 41, 51, 61, 71, 81, 91 )
    // CHECK-NEXT: ( 27, 41, 55, 69, 83, 97, 111, 125 )
    // CHECK-NEXT: ( 33, 51, 69, 87, 105, 123, 141, 159 )
    // CHECK-NEXT: ( 39, 61, 83, 105, 127, 149, 171, 193 )
    // CHECK-NEXT: ( 45, 71, 97, 123, 149, 175, 201, 227 )
    // CHECK-NEXT: ( 51, 81, 111, 141, 171, 201, 231, 261 )
    // CHECK-NEXT: ( 57, 91, 125, 159, 193, 227, 261, 295 )
    //
    call @dump(%0) : (tensor<8x8xf32>) -> ()
    call @dump(%1) : (tensor<8x8xf32>) -> ()
    call @dump(%2) : (tensor<8x8xf32>) -> ()
    call @dump(%3) : (tensor<8x8xf32>) -> ()
    call @dump(%4) : (tensor<8x8xf32>) -> ()
    call @dump(%5) : (tensor<8x8xf32>) -> ()
    call @dump(%6) : (tensor<8x8xf32>) -> ()
    call @dump(%7) : (tensor<8x8xf32>) -> ()
    call @dump(%8) : (tensor<8x8xf32>) -> ()
    call @dump(%9) : (tensor<8x8xf32>) -> ()
    call @dump(%10) : (tensor<8x8xf32>) -> ()
    call @dump(%11) : (tensor<8x8xf32>) -> ()
    call @dump(%12) : (tensor<8x8xf32>) -> ()
    call @dump(%13) : (tensor<8x8xf32>) -> ()
    call @dump(%14) : (tensor<8x8xf32>) -> ()
    call @dump(%15) : (tensor<8x8xf32>) -> ()
    call @dump(%16) : (tensor<8x8xf32>) -> ()
    call @dump(%17) : (tensor<8x8xf32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %Acoo : tensor<8x4xf32, #SortedCOO>
    bufferization.dealloc_tensor %Acsr : tensor<8x4xf32, #CSR>
    bufferization.dealloc_tensor %Acsc : tensor<8x4xf32, #CSC>
    bufferization.dealloc_tensor %ATcoo : tensor<4x8xf32, #SortedCOO>
    bufferization.dealloc_tensor %ATcsr : tensor<4x8xf32, #CSR>
    bufferization.dealloc_tensor %ATcsc : tensor<4x8xf32, #CSC>

    llvm.call @mgpuDestroySparseEnv() : () -> ()

    return
  }
}
