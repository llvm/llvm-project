// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
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
  func @matmul1(%A: tensor<4x8xf64>,
                %B: tensor<8x4xf64>) -> tensor<4x4xf64> {
    %C = arith.constant dense<0.0> : tensor<4x4xf64>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64>, tensor<8x4xf64>)
         outs(%C: tensor<4x4xf64>) -> tensor<4x4xf64>
    return %D: tensor<4x4xf64>
  }

  //
  // Computes C = A x B with all matrices sparse (SpMSpM) in CSR.
  //
  func @matmul2(%A: tensor<4x8xf64, #CSR>,
                %B: tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR> {
    %c4 = arith.constant 4 : index
    %C = sparse_tensor.init [%c4, %c4] : tensor<4x4xf64, #CSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #CSR>, tensor<8x4xf64, #CSR>)
         outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    return %D: tensor<4x4xf64, #CSR>
  }

  //
  // Computes C = A x B with all matrices sparse (SpMSpM) in DCSR.
  //
  func @matmul3(%A: tensor<4x8xf64, #DCSR>,
                %B: tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %c4 = arith.constant 4 : index
    %C = sparse_tensor.init [%c4, %c4] : tensor<4x4xf64, #DCSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #DCSR>, tensor<8x4xf64, #DCSR>)
         outs(%C: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    return %D: tensor<4x4xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func @entry() {
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

    // Convert all these matrices to sparse format.
    %a1 = sparse_tensor.convert %da : tensor<4x8xf64> to tensor<4x8xf64, #CSR>
    %a2 = sparse_tensor.convert %da : tensor<4x8xf64> to tensor<4x8xf64, #DCSR>
    %a3 = sparse_tensor.convert %sa : tensor<4x8xf64> to tensor<4x8xf64, #CSR>
    %a4 = sparse_tensor.convert %sa : tensor<4x8xf64> to tensor<4x8xf64, #DCSR>
    %b1 = sparse_tensor.convert %db : tensor<8x4xf64> to tensor<8x4xf64, #CSR>
    %b2 = sparse_tensor.convert %db : tensor<8x4xf64> to tensor<8x4xf64, #DCSR>
    %b3 = sparse_tensor.convert %sb : tensor<8x4xf64> to tensor<8x4xf64, #CSR>
    %b4 = sparse_tensor.convert %sb : tensor<8x4xf64> to tensor<8x4xf64, #DCSR>

    // Call kernels with dense.
    %0 = call @matmul1(%da, %db)
       : (tensor<4x8xf64>, tensor<8x4xf64>) -> tensor<4x4xf64>
    %1 = call @matmul2(%a1, %b1)
       : (tensor<4x8xf64, #CSR>,
          tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    %2 = call @matmul3(%a2, %b2)
       : (tensor<4x8xf64, #DCSR>,
          tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    // Call kernels with one sparse.
    %3 = call @matmul1(%sa, %db)
       : (tensor<4x8xf64>, tensor<8x4xf64>) -> tensor<4x4xf64>
    %4 = call @matmul2(%a3, %b1)
       : (tensor<4x8xf64, #CSR>,
          tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    %5 = call @matmul3(%a4, %b2)
       : (tensor<4x8xf64, #DCSR>,
          tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    // Call kernels with sparse.
    %6 = call @matmul1(%sa, %sb)
       : (tensor<4x8xf64>, tensor<8x4xf64>) -> tensor<4x4xf64>
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
    %m0 = bufferization.to_memref %0 : memref<4x4xf64>
    %v0 = vector.transfer_read %m0[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v0 : vector<4x4xf64>

    //
    // CHECK:    ( ( 388.76, 425.56, 462.36, 499.16 ),
    // CHECK-SAME: ( 397.12, 434.72, 472.32, 509.92 ),
    // CHECK-SAME: ( 405.48, 443.88, 482.28, 520.68 ),
    // CHECK-SAME: ( 413.84, 453.04, 492.24, 531.44 ) )
    //
    %c1 = sparse_tensor.convert %1 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %m1 = bufferization.to_memref %c1 : memref<4x4xf64>
    %v1 = vector.transfer_read %m1[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v1 : vector<4x4xf64>

    //
    // CHECK:    ( ( 388.76, 425.56, 462.36, 499.16 ),
    // CHECK-SAME: ( 397.12, 434.72, 472.32, 509.92 ),
    // CHECK-SAME: ( 405.48, 443.88, 482.28, 520.68 ),
    // CHECK-SAME: ( 413.84, 453.04, 492.24, 531.44 ) )
    //
    %c2 = sparse_tensor.convert %2 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %m2 = bufferization.to_memref %c2 : memref<4x4xf64>
    %v2 = vector.transfer_read %m2[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v2 : vector<4x4xf64>

    //
    // CHECK:    ( ( 86.08, 94.28, 102.48, 110.68 ),
    // CHECK-SAME: ( 0, 0, 0, 0 ),
    // CHECK-SAME: ( 23.46, 25.76, 28.06, 30.36 ),
    // CHECK-SAME: ( 10.8, 11.8, 12.8, 13.8 ) )
    //
    %m3 = bufferization.to_memref %3 : memref<4x4xf64>
    %v3 = vector.transfer_read %m3[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v3 : vector<4x4xf64>

    //
    // CHECK:    ( ( 86.08, 94.28, 102.48, 110.68 ),
    // CHECK-SAME: ( 0, 0, 0, 0 ),
    // CHECK-SAME: ( 23.46, 25.76, 28.06, 30.36 ),
    // CHECK-SAME: ( 10.8, 11.8, 12.8, 13.8 ) )
    //
    %c4 = sparse_tensor.convert %4 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %m4 = bufferization.to_memref %c4 : memref<4x4xf64>
    %v4 = vector.transfer_read %m4[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v4 : vector<4x4xf64>

    //
    // CHECK:    ( ( 86.08, 94.28, 102.48, 110.68 ),
    // CHECK-SAME: ( 0, 0, 0, 0 ),
    // CHECK-SAME: ( 23.46, 25.76, 28.06, 30.36 ),
    // CHECK-SAME: ( 10.8, 11.8, 12.8, 13.8 ) )
    //
    %c5 = sparse_tensor.convert %5 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %m5 = bufferization.to_memref %c5 : memref<4x4xf64>
    %v5 = vector.transfer_read %m5[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v5 : vector<4x4xf64>

    //
    // CHECK: ( ( 0, 30.5, 4.2, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 4.6, 0 ), ( 0, 0, 7, 8 ) )
    //
    %m6 = bufferization.to_memref %6 : memref<4x4xf64>
    %v6 = vector.transfer_read %m6[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v6 : vector<4x4xf64>

    //
    // CHECK: ( ( 0, 30.5, 4.2, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 4.6, 0 ), ( 0, 0, 7, 8 ) )
    //
    %c7 = sparse_tensor.convert %7 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %m7 = bufferization.to_memref %c7 : memref<4x4xf64>
    %v7 = vector.transfer_read %m7[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v7 : vector<4x4xf64>

    //
    // CHECK: ( ( 0, 30.5, 4.2, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 4.6, 0 ), ( 0, 0, 7, 8 ) )
    //
    %c8 = sparse_tensor.convert %8 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %m8 = bufferization.to_memref %c8 : memref<4x4xf64>
    %v8 = vector.transfer_read %m8[%c0, %c0], %d1 : memref<4x4xf64>, vector<4x4xf64>
    vector.print %v8 : vector<4x4xf64>

    //
    // Sanity check on nonzeros.
    //
    // CHECK: ( 30.5, 4.2, 4.6, 7, 8, -1, -1, -1 )
    // CHECK: ( 30.5, 4.2, 4.6, 7, 8, -1, -1, -1 )
    //
    %val7 = sparse_tensor.values %7 : tensor<4x4xf64, #CSR> to memref<?xf64>
    %val8 = sparse_tensor.values %8 : tensor<4x4xf64, #DCSR> to memref<?xf64>
    %nz7 = vector.transfer_read %val7[%c0], %d1 : memref<?xf64>, vector<8xf64>
    %nz8 = vector.transfer_read %val8[%c0], %d1 : memref<?xf64>, vector<8xf64>
    vector.print %nz7 : vector<8xf64>
    vector.print %nz8 : vector<8xf64>

    // Release the resources.
    sparse_tensor.release %a1 : tensor<4x8xf64, #CSR>
    sparse_tensor.release %a2 : tensor<4x8xf64, #DCSR>
    sparse_tensor.release %a3 : tensor<4x8xf64, #CSR>
    sparse_tensor.release %a4 : tensor<4x8xf64, #DCSR>
    sparse_tensor.release %b1 : tensor<8x4xf64, #CSR>
    sparse_tensor.release %b2 : tensor<8x4xf64, #DCSR>
    sparse_tensor.release %b3 : tensor<8x4xf64, #CSR>
    sparse_tensor.release %b4 : tensor<8x4xf64, #DCSR>
    sparse_tensor.release %1 : tensor<4x4xf64, #CSR>
    sparse_tensor.release %2 : tensor<4x4xf64, #DCSR>
    sparse_tensor.release %4 : tensor<4x4xf64, #CSR>
    sparse_tensor.release %5 : tensor<4x4xf64, #DCSR>
    sparse_tensor.release %7 : tensor<4x4xf64, #CSR>
    sparse_tensor.release %8 : tensor<4x4xf64, #DCSR>
    memref.dealloc %m0 : memref<4x4xf64>
    memref.dealloc %m1 : memref<4x4xf64>
    memref.dealloc %m2 : memref<4x4xf64>
    memref.dealloc %m3 : memref<4x4xf64>
    memref.dealloc %m4 : memref<4x4xf64>
    memref.dealloc %m5 : memref<4x4xf64>
    memref.dealloc %m6 : memref<4x4xf64>
    memref.dealloc %m7 : memref<4x4xf64>
    memref.dealloc %m8 : memref<4x4xf64>

    return
  }
}
