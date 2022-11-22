// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with parallelization.
//
// RUN: mlir-opt %s --sparse-compiler="parallelization-strategy=any-storage-any-loop" | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with direct IR generation.
//
// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

module {
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
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #CSR>
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
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #DCSR>, tensor<8x4xf64, #DCSR>)
         outs(%C: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    return %D: tensor<4x4xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f64

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
    // Sanity check on stored entries before going into the computations.
    //
    // CHECK:      32
    // CHECK-NEXT: 32
    // CHECK-NEXT: 4
    // CHECK-NEXT: 4
    // CHECK-NEXT: 32
    // CHECK-NEXT: 32
    // CHECK-NEXT: 8
    // CHECK-NEXT: 8
    //
    %noea1 = sparse_tensor.number_of_entries %a1 : tensor<4x8xf64, #CSR>
    %noea2 = sparse_tensor.number_of_entries %a2 : tensor<4x8xf64, #DCSR>
    %noea3 = sparse_tensor.number_of_entries %a3 : tensor<4x8xf64, #CSR>
    %noea4 = sparse_tensor.number_of_entries %a4 : tensor<4x8xf64, #DCSR>
    %noeb1 = sparse_tensor.number_of_entries %b1 : tensor<8x4xf64, #CSR>
    %noeb2 = sparse_tensor.number_of_entries %b2 : tensor<8x4xf64, #DCSR>
    %noeb3 = sparse_tensor.number_of_entries %b3 : tensor<8x4xf64, #CSR>
    %noeb4 = sparse_tensor.number_of_entries %b4 : tensor<8x4xf64, #DCSR>
    vector.print %noea1 : index
    vector.print %noea2 : index
    vector.print %noea3 : index
    vector.print %noea4 : index
    vector.print %noeb1 : index
    vector.print %noeb2 : index
    vector.print %noeb3 : index
    vector.print %noeb4 : index

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
    // CHECK:    ( ( 388.76, 425.56, 462.36, 499.16 ),
    // CHECK-SAME: ( 397.12, 434.72, 472.32, 509.92 ),
    // CHECK-SAME: ( 405.48, 443.88, 482.28, 520.68 ),
    // CHECK-SAME: ( 413.84, 453.04, 492.24, 531.44 ) )
    //
    %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v0 : vector<4x4xf64>

    //
    // CHECK:    ( ( 388.76, 425.56, 462.36, 499.16 ),
    // CHECK-SAME: ( 397.12, 434.72, 472.32, 509.92 ),
    // CHECK-SAME: ( 405.48, 443.88, 482.28, 520.68 ),
    // CHECK-SAME: ( 413.84, 453.04, 492.24, 531.44 ) )
    //
    %c1 = sparse_tensor.convert %1 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %v1 = vector.transfer_read %c1[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v1 : vector<4x4xf64>

    //
    // CHECK:    ( ( 388.76, 425.56, 462.36, 499.16 ),
    // CHECK-SAME: ( 397.12, 434.72, 472.32, 509.92 ),
    // CHECK-SAME: ( 405.48, 443.88, 482.28, 520.68 ),
    // CHECK-SAME: ( 413.84, 453.04, 492.24, 531.44 ) )
    //
    %c2 = sparse_tensor.convert %2 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %v2 = vector.transfer_read %c2[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v2 : vector<4x4xf64>

    //
    // CHECK:    ( ( 86.08, 94.28, 102.48, 110.68 ),
    // CHECK-SAME: ( 0, 0, 0, 0 ),
    // CHECK-SAME: ( 23.46, 25.76, 28.06, 30.36 ),
    // CHECK-SAME: ( 10.8, 11.8, 12.8, 13.8 ) )
    //
    %v3 = vector.transfer_read %3[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v3 : vector<4x4xf64>

    //
    // CHECK:    ( ( 86.08, 94.28, 102.48, 110.68 ),
    // CHECK-SAME: ( 0, 0, 0, 0 ),
    // CHECK-SAME: ( 23.46, 25.76, 28.06, 30.36 ),
    // CHECK-SAME: ( 10.8, 11.8, 12.8, 13.8 ) )
    //
    %c4 = sparse_tensor.convert %4 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %v4 = vector.transfer_read %c4[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v4 : vector<4x4xf64>

    //
    // CHECK:    ( ( 86.08, 94.28, 102.48, 110.68 ),
    // CHECK-SAME: ( 0, 0, 0, 0 ),
    // CHECK-SAME: ( 23.46, 25.76, 28.06, 30.36 ),
    // CHECK-SAME: ( 10.8, 11.8, 12.8, 13.8 ) )
    //
    %c5 = sparse_tensor.convert %5 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %v5 = vector.transfer_read %c5[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v5 : vector<4x4xf64>

    //
    // CHECK-NEXT: ( ( 0, 30.5, 4.2, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 4.6, 0 ), ( 0, 0, 7, 8 ) )
    //
    %v6 = vector.transfer_read %6[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v6 : vector<4x4xf64>

    //
    // CHECK-NEXT: ( ( 0, 30.5, 4.2, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 4.6, 0 ), ( 0, 0, 7, 8 ) )
    //
    %c7 = sparse_tensor.convert %7 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %v7 = vector.transfer_read %c7[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v7 : vector<4x4xf64>

    //
    // CHECK-NEXT: ( ( 0, 30.5, 4.2, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 4.6, 0 ), ( 0, 0, 7, 8 ) )
    //
    %c8 = sparse_tensor.convert %8 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %v8 = vector.transfer_read %c8[%c0, %c0], %d1 : tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v8 : vector<4x4xf64>

    //
    // Sanity check on nonzeros.
    //
    // CHECK-NEXT: ( 30.5, 4.2, 4.6, 7, 8 )
    // CHECK-NEXT: ( 30.5, 4.2, 4.6, 7, 8 )
    //
    %val7 = sparse_tensor.values %7 : tensor<4x4xf64, #CSR> to memref<?xf64>
    %val8 = sparse_tensor.values %8 : tensor<4x4xf64, #DCSR> to memref<?xf64>
    %nz7 = vector.transfer_read %val7[%c0], %d1 : memref<?xf64>, vector<5xf64>
    %nz8 = vector.transfer_read %val8[%c0], %d1 : memref<?xf64>, vector<5xf64>
    vector.print %nz7 : vector<5xf64>
    vector.print %nz8 : vector<5xf64>

    //
    // Sanity check on stored entries after the computations.
    //
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    //
    %noe7 = sparse_tensor.number_of_entries %7 : tensor<4x4xf64, #CSR>
    %noe8 = sparse_tensor.number_of_entries %8 : tensor<4x4xf64, #DCSR>
    vector.print %noe7 : index
    vector.print %noe8 : index

    // Release the resources.
    bufferization.dealloc_tensor %a1 : tensor<4x8xf64, #CSR>
    bufferization.dealloc_tensor %a2 : tensor<4x8xf64, #DCSR>
    bufferization.dealloc_tensor %a3 : tensor<4x8xf64, #CSR>
    bufferization.dealloc_tensor %a4 : tensor<4x8xf64, #DCSR>
    bufferization.dealloc_tensor %b1 : tensor<8x4xf64, #CSR>
    bufferization.dealloc_tensor %b2 : tensor<8x4xf64, #DCSR>
    bufferization.dealloc_tensor %b3 : tensor<8x4xf64, #CSR>
    bufferization.dealloc_tensor %b4 : tensor<8x4xf64, #DCSR>
    bufferization.dealloc_tensor %1 : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %2 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %4 : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %5 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %7 : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %8 : tensor<4x4xf64, #DCSR>

    return
  }
}
