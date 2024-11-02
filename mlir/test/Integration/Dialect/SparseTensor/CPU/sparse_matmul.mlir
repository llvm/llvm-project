//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with parallelization strategy.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=true parallelization-strategy=any-storage-any-loop
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and parallelization strategy.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true parallelization-strategy=any-storage-any-loop
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

// TODO: Investigate the output generated for SVE, see https://github.com/llvm/llvm-project/issues/60626

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

module {
  func.func private @printMemrefF64(%ptr : tensor<*xf64>)
  func.func private @printMemref1dF64(%ptr : memref<?xf64>) attributes { llvm.emit_c_interface }

  //
  // Computes C = A x B with all matrices dense.
  //
  func.func @matmul1(%A: tensor<4x8xf64>, %B: tensor<8x4xf64>,
                     %C: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64>, tensor<8x4xf64>)
      outs(%C: tensor<4x4xf64>) -> tensor<4x4xf64>
    return %D: tensor<4x4xf64>
  }

  //
  // Computes C = A x B with all matrices sparse (SpMSpM) in CSR.
  //
  func.func @matmul2(%A: tensor<4x8xf64, #CSR>,
                     %B: tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR> {
    %C = tensor.empty() : tensor<4x4xf64, #CSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #CSR>, tensor<8x4xf64, #CSR>)
         outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    return %D: tensor<4x4xf64, #CSR>
  }

  //
  // Computes C = A x B with all matrices sparse (SpMSpM) in DCSR.
  //
  func.func @matmul3(%A: tensor<4x8xf64, #DCSR>,
                     %B: tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = tensor.empty() : tensor<4x4xf64, #DCSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #DCSR>, tensor<8x4xf64, #DCSR>)
         outs(%C: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    return %D: tensor<4x4xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index

    // Initialize various matrices, dense for stress testing,
    // and sparse to verify correct nonzero structure.
    %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
    ]> : tensor<4x8xf64>
    %db = arith.constant dense<[
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ],
        [ 10.3, 11.3, 12.3, 13.3 ],
        [ 10.4, 11.4, 12.4, 13.4 ],
        [ 10.5, 11.5, 12.5, 13.5 ],
        [ 10.6, 11.6, 12.6, 13.6 ],
        [ 10.7, 11.7, 12.7, 13.7 ],
        [ 10.8, 11.8, 12.8, 13.8 ]
    ]> : tensor<8x4xf64>
    %sa = arith.constant dense<[
        [ 0.0, 2.1, 0.0, 0.0, 0.0, 6.1, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
    ]> : tensor<4x8xf64>
    %sb = arith.constant dense<[
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 0.0, 0.0, 2.0, 0.0 ],
        [ 0.0, 3.0, 0.0, 0.0 ],
        [ 4.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 5.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 6.0, 0.0 ],
        [ 0.0, 0.0, 7.0, 8.0 ]
    ]> : tensor<8x4xf64>
    %zero = arith.constant dense<0.0> : tensor<4x4xf64>

    // Convert all these matrices to sparse format.
    %a1 = sparse_tensor.convert %da : tensor<4x8xf64> to tensor<4x8xf64, #CSR>
    %a2 = sparse_tensor.convert %da : tensor<4x8xf64> to tensor<4x8xf64, #DCSR>
    %a3 = sparse_tensor.convert %sa : tensor<4x8xf64> to tensor<4x8xf64, #CSR>
    %a4 = sparse_tensor.convert %sa : tensor<4x8xf64> to tensor<4x8xf64, #DCSR>
    %b1 = sparse_tensor.convert %db : tensor<8x4xf64> to tensor<8x4xf64, #CSR>
    %b2 = sparse_tensor.convert %db : tensor<8x4xf64> to tensor<8x4xf64, #DCSR>
    %b3 = sparse_tensor.convert %sb : tensor<8x4xf64> to tensor<8x4xf64, #CSR>
    %b4 = sparse_tensor.convert %sb : tensor<8x4xf64> to tensor<8x4xf64, #DCSR>

    //
    // Sanity check before going into the computations.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: pos[1] : ( 0, 8, 16, 24, 32
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
    // CHECK-NEXT: values : ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %a1 : tensor<4x8xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: pos[0] : ( 0, 4
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 8, 16, 24, 32
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
    // CHECK-NEXT: values : ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %a2 : tensor<4x8xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: pos[1] : ( 0, 2, 2, 3, 4
    // CHECK-NEXT: crd[1] : ( 1, 5, 1, 7
    // CHECK-NEXT: values : ( 2.1, 6.1, 2.3, 1
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %a3 : tensor<4x8xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 4
    // CHECK-NEXT: crd[1] : ( 1, 5, 1, 7
    // CHECK-NEXT: values : ( 2.1, 6.1, 2.3, 1
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %a4 : tensor<4x8xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12, 16, 20, 24, 28, 32
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 10.1, 11.1, 12.1, 13.1, 10.2, 11.2, 12.2, 13.2, 10.3, 11.3, 12.3, 13.3, 10.4, 11.4, 12.4, 13.4, 10.5, 11.5, 12.5, 13.5, 10.6, 11.6, 12.6, 13.6, 10.7, 11.7, 12.7, 13.7, 10.8, 11.8, 12.8, 13.8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %b1 : tensor<8x4xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: pos[0] : ( 0, 8
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5, 6, 7
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12, 16, 20, 24, 28, 32
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 10.1, 11.1, 12.1, 13.1, 10.2, 11.2, 12.2, 13.2, 10.3, 11.3, 12.3, 13.3, 10.4, 11.4, 12.4, 13.4, 10.5, 11.5, 12.5, 13.5, 10.6, 11.6, 12.6, 13.6, 10.7, 11.7, 12.7, 13.7, 10.8, 11.8, 12.8, 13.8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %b2 : tensor<8x4xf64, #DCSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3, 4, 4, 5, 6, 8
    // CHECK-NEXT: crd[1] : ( 3, 2, 1, 0, 1, 2, 2, 3
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %b3 : tensor<8x4xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: pos[0] : ( 0, 7
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 5, 6, 7
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3, 4, 5, 6, 8
    // CHECK-NEXT: crd[1] : ( 3, 2, 1, 0, 1, 2, 2, 3
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %b4 : tensor<8x4xf64, #DCSR>

    // Call kernels with dense.
    %0 = call @matmul1(%da, %db, %zero)
       : (tensor<4x8xf64>, tensor<8x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = call @matmul2(%a1, %b1)
       : (tensor<4x8xf64, #CSR>,
          tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    %2 = call @matmul3(%a2, %b2)
       : (tensor<4x8xf64, #DCSR>,
          tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    // Call kernels with one sparse.
    %3 = call @matmul1(%sa, %db, %zero)
       : (tensor<4x8xf64>, tensor<8x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    %4 = call @matmul2(%a3, %b1)
       : (tensor<4x8xf64, #CSR>,
          tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    %5 = call @matmul3(%a4, %b2)
       : (tensor<4x8xf64, #DCSR>,
          tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    // Call kernels with sparse.
    %6 = call @matmul1(%sa, %sb, %zero)
       : (tensor<4x8xf64>, tensor<8x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    %7 = call @matmul2(%a3, %b3)
       : (tensor<4x8xf64, #CSR>,
          tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    %8 = call @matmul3(%a4, %b4)
       : (tensor<4x8xf64, #DCSR>,
          tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    //
    // CHECK:      {{\[}}[388.76,   425.56,   462.36,   499.16],
    // CHECK-NEXT: [397.12,   434.72,   472.32,   509.92],
    // CHECK-NEXT: [405.48,   443.88,   482.28,   520.68],
    // CHECK-NEXT: [413.84,   453.04,   492.24,   531.44]]
    //
    %u0 = tensor.cast %0 : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%u0) : (tensor<*xf64>) -> ()

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 16
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12, 16
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 388.76, 425.56, 462.36, 499.16, 397.12, 434.72, 472.32, 509.92, 405.48, 443.88, 482.28, 520.68, 413.84, 453.04, 492.24, 531.44
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<4x4xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 16
    // CHECK-NEXT: pos[0] : ( 0, 4
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12, 16
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 388.76, 425.56, 462.36, 499.16, 397.12, 434.72, 472.32, 509.92, 405.48, 443.88, 482.28, 520.68, 413.84, 453.04, 492.24, 531.44
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %2 : tensor<4x4xf64, #DCSR>

    //
    // CHECK:      {{\[}}[86.08,   94.28,   102.48,   110.68],
    // CHECK-NEXT: [0,   0,   0,   0],
    // CHECK-NEXT: [23.46,   25.76,   28.06,   30.36],
    // CHECK-NEXT: [10.8,   11.8,   12.8,   13.8]]
    //
    %u3 = tensor.cast %3 : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%u3) : (tensor<*xf64>) -> ()

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: pos[1] : ( 0, 4, 4, 8, 12
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 86.08, 94.28, 102.48, 110.68, 23.46, 25.76, 28.06, 30.36, 10.8, 11.8, 12.8, 13.8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %4 : tensor<4x4xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 86.08, 94.28, 102.48, 110.68, 23.46, 25.76, 28.06, 30.36, 10.8, 11.8, 12.8, 13.8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %5 : tensor<4x4xf64, #DCSR>

    //
    // CHECK:      {{\[}}[0,   30.5,   4.2,   0],
    // CHECK-NEXT: [0,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   4.6,   0],
    // CHECK-NEXT: [0,   0,   7,   8]]
    //
    %u6 = tensor.cast %6 : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%u6) : (tensor<*xf64>) -> ()

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: pos[1] : ( 0, 2, 2, 3, 5
    // CHECK-NEXT: crd[1] : ( 1, 2, 2, 2, 3
    // CHECK-NEXT: values : ( 30.5, 4.2, 4.6, 7, 8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %7 : tensor<4x4xf64, #CSR>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 5
    // CHECK-NEXT: crd[1] : ( 1, 2, 2, 2, 3
    // CHECK-NEXT: values : ( 30.5, 4.2, 4.6, 7, 8
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %8 : tensor<4x4xf64, #DCSR>

    // Release the resources.
    bufferization.dealloc_tensor %a1 : tensor<4x8xf64, #CSR>
    bufferization.dealloc_tensor %a2 : tensor<4x8xf64, #DCSR>
    bufferization.dealloc_tensor %a3 : tensor<4x8xf64, #CSR>
    bufferization.dealloc_tensor %a4 : tensor<4x8xf64, #DCSR>
    bufferization.dealloc_tensor %b1 : tensor<8x4xf64, #CSR>
    bufferization.dealloc_tensor %b2 : tensor<8x4xf64, #DCSR>
    bufferization.dealloc_tensor %b3 : tensor<8x4xf64, #CSR>
    bufferization.dealloc_tensor %b4 : tensor<8x4xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<4x4xf64>
    bufferization.dealloc_tensor %1 : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %2 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<4x4xf64>
    bufferization.dealloc_tensor %4 : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %5 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %6 : tensor<4x4xf64>
    bufferization.dealloc_tensor %7 : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %8 : tensor<4x4xf64, #DCSR>

    return
  }
}
