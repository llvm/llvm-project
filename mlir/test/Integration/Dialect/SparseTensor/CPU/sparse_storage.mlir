// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

//
// Several common sparse storage schemes.
//

#Dense  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense" ]
}>

#CSR  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

#DCSC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

#BlockRow = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ]
}>

#BlockCol = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

//
// Integration test that looks "under the hood" of sparse storage schemes.
//
module {
  //
  // Main driver that initializes a sparse tensor and inspects the sparse
  // storage schemes in detail. Note that users of the MLIR sparse compiler
  // are typically not concerned with such details, but the test ensures
  // everything is working "under the hood".
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = arith.constant 0.0 : f64

    //
    // Initialize a dense tensor.
    //
    %t = arith.constant dense<[
       [ 1.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  3.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  5.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  0.0,  6.0,  0.0,  0.0,  0.0],
       [ 0.0,  7.0,  8.0,  0.0,  0.0,  0.0,  0.0,  9.0],
       [ 0.0,  0.0, 10.0,  0.0,  0.0,  0.0, 11.0, 12.0],
       [ 0.0, 13.0, 14.0,  0.0,  0.0,  0.0, 15.0, 16.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 17.0,  0.0]
    ]> : tensor<10x8xf64>

    //
    // Convert dense tensor to various sparse tensors.
    //
    %0 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #Dense>
    %1 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #CSR>
    %2 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #DCSR>
    %3 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #CSC>
    %4 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #DCSC>
    %x = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #BlockRow>
    %y = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #BlockCol>

    //
    // Inspect storage scheme of Dense.
    //
    // CHECK:    ( 1, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME: 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
    // CHECK-SAME: 0, 0, 0, 0, 6, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 9,
    // CHECK-SAME: 0, 0, 10, 0, 0, 0, 11, 12, 0, 13, 14, 0, 0, 0, 15, 16,
    // CHECK-SAME: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0 )
    //
    %5 = sparse_tensor.values %0 : tensor<10x8xf64, #Dense> to memref<?xf64>
    %6 = vector.transfer_read %5[%c0], %d0: memref<?xf64>, vector<80xf64>
    vector.print %6 : vector<80xf64>

    //
    // Inspect storage scheme of CSR.
    //
    // pointers(1)
    // indices(1)
    // values
    //
    // CHECK: ( 0, 3, 3, 4, 5, 6, 9, 12, 16, 16, 17 )
    // CHECK: ( 0, 2, 7, 2, 3, 4, 1, 2, 7, 2, 6, 7, 1, 2, 6, 7, 6 )
    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 )
    //
    %7 = sparse_tensor.pointers %1 { dimension = 1 : index } : tensor<10x8xf64, #CSR> to memref<?xindex>
    %8 = vector.transfer_read %7[%c0], %c0: memref<?xindex>, vector<11xindex>
    vector.print %8 : vector<11xindex>
    %9 = sparse_tensor.indices %1 { dimension = 1 : index } : tensor<10x8xf64, #CSR> to memref<?xindex>
    %10 = vector.transfer_read %9[%c0], %c0: memref<?xindex>, vector<17xindex>
    vector.print %10 : vector<17xindex>
    %11 = sparse_tensor.values %1 : tensor<10x8xf64, #CSR> to memref<?xf64>
    %12 = vector.transfer_read %11[%c0], %d0: memref<?xf64>, vector<17xf64>
    vector.print %12 : vector<17xf64>

    //
    // Inspect storage scheme of DCSR.
    //
    // pointers(0)
    // indices(0)
    // pointers(1)
    // indices(1)
    // values
    //
    // CHECK: ( 0, 8 )
    // CHECK: ( 0, 2, 3, 4, 5, 6, 7, 9 )
    // CHECK: ( 0, 3, 4, 5, 6, 9, 12, 16, 17 )
    // CHECK: ( 0, 2, 7, 2, 3, 4, 1, 2, 7, 2, 6, 7, 1, 2, 6, 7, 6 )
    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 )
    //
    %13 = sparse_tensor.pointers %2 { dimension = 0 : index } : tensor<10x8xf64, #DCSR> to memref<?xindex>
    %14 = vector.transfer_read %13[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %14 : vector<2xindex>
    %15 = sparse_tensor.indices %2 { dimension = 0 : index } : tensor<10x8xf64, #DCSR> to memref<?xindex>
    %16 = vector.transfer_read %15[%c0], %c0: memref<?xindex>, vector<8xindex>
    vector.print %16 : vector<8xindex>
    %17 = sparse_tensor.pointers %2 { dimension = 1 : index } : tensor<10x8xf64, #DCSR> to memref<?xindex>
    %18 = vector.transfer_read %17[%c0], %c0: memref<?xindex>, vector<9xindex>
    vector.print %18 : vector<9xindex>
    %19 = sparse_tensor.indices %2 { dimension = 1 : index } : tensor<10x8xf64, #DCSR> to memref<?xindex>
    %20 = vector.transfer_read %19[%c0], %c0: memref<?xindex>, vector<17xindex>
    vector.print %20 : vector<17xindex>
    %21 = sparse_tensor.values %2 : tensor<10x8xf64, #DCSR> to memref<?xf64>
    %22 = vector.transfer_read %21[%c0], %d0: memref<?xf64>, vector<17xf64>
    vector.print %22 : vector<17xf64>

    //
    // Inspect storage scheme of CSC.
    //
    // pointers(1)
    // indices(1)
    // values
    //
    // CHECK: ( 0, 1, 3, 8, 9, 10, 10, 13, 17 )
    // CHECK: ( 0, 5, 7, 0, 2, 5, 6, 7, 3, 4, 6, 7, 9, 0, 5, 6, 7 )
    // CHECK: ( 1, 7, 13, 2, 4, 8, 10, 14, 5, 6, 11, 15, 17, 3, 9, 12, 16 )
    //
    %23 = sparse_tensor.pointers %3 { dimension = 1 : index } : tensor<10x8xf64, #CSC> to memref<?xindex>
    %24 = vector.transfer_read %23[%c0], %c0: memref<?xindex>, vector<9xindex>
    vector.print %24 : vector<9xindex>
    %25 = sparse_tensor.indices %3 { dimension = 1 : index } : tensor<10x8xf64, #CSC> to memref<?xindex>
    %26 = vector.transfer_read %25[%c0], %c0: memref<?xindex>, vector<17xindex>
    vector.print %26 : vector<17xindex>
    %27 = sparse_tensor.values %3 : tensor<10x8xf64, #CSC> to memref<?xf64>
    %28 = vector.transfer_read %27[%c0], %d0: memref<?xf64>, vector<17xf64>
    vector.print %28 : vector<17xf64>

    //
    // Inspect storage scheme of DCSC.
    //
    // pointers(0)
    // indices(0)
    // pointers(1)
    // indices(1)
    // values
    //
    // CHECK: ( 0, 7 )
    // CHECK: ( 0, 1, 2, 3, 4, 6, 7 )
    // CHECK: ( 0, 1, 3, 8, 9, 10, 13, 17 )
    // CHECK: ( 0, 5, 7, 0, 2, 5, 6, 7, 3, 4, 6, 7, 9, 0, 5, 6, 7 )
    // CHECK: ( 1, 7, 13, 2, 4, 8, 10, 14, 5, 6, 11, 15, 17, 3, 9, 12, 16 )
    //
    %29 = sparse_tensor.pointers %4 { dimension = 0 : index } : tensor<10x8xf64, #DCSC> to memref<?xindex>
    %30 = vector.transfer_read %29[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %30 : vector<2xindex>
    %31 = sparse_tensor.indices %4 { dimension = 0 : index } : tensor<10x8xf64, #DCSC> to memref<?xindex>
    %32 = vector.transfer_read %31[%c0], %c0: memref<?xindex>, vector<7xindex>
    vector.print %32 : vector<7xindex>
    %33 = sparse_tensor.pointers %4 { dimension = 1 : index } : tensor<10x8xf64, #DCSC> to memref<?xindex>
    %34 = vector.transfer_read %33[%c0], %c0: memref<?xindex>, vector<8xindex>
    vector.print %34 : vector<8xindex>
    %35 = sparse_tensor.indices %4 { dimension = 1 : index } : tensor<10x8xf64, #DCSC> to memref<?xindex>
    %36 = vector.transfer_read %35[%c0], %c0: memref<?xindex>, vector<17xindex>
    vector.print %36 : vector<17xindex>
    %37 = sparse_tensor.values %4 : tensor<10x8xf64, #DCSC> to memref<?xf64>
    %38 = vector.transfer_read %37[%c0], %d0: memref<?xf64>, vector<17xf64>
    vector.print %38 : vector<17xf64>

    //
    // Inspect storage scheme of BlockRow.
    //
    // pointers(0)
    // indices(0)
    // values
    //
    // CHECK: ( 0, 8 )
    // CHECK: ( 0, 2, 3, 4, 5, 6, 7, 9 )
    // CHECK: ( 1, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0,
    // CHECK-SAME: 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0,
    // CHECK-SAME: 0, 7, 8, 0, 0, 0, 0, 9, 0, 0, 10, 0, 0, 0, 11, 12,
    // CHECK-SAME: 0, 13, 14, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 17, 0 )
    //
    %39 = sparse_tensor.pointers %x { dimension = 0 : index } : tensor<10x8xf64, #BlockRow> to memref<?xindex>
    %40 = vector.transfer_read %39[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %40 : vector<2xindex>
    %41 = sparse_tensor.indices %x { dimension = 0 : index } : tensor<10x8xf64, #BlockRow> to memref<?xindex>
    %42 = vector.transfer_read %41[%c0], %c0: memref<?xindex>, vector<8xindex>
    vector.print %42 : vector<8xindex>
    %43 = sparse_tensor.values %x : tensor<10x8xf64, #BlockRow> to memref<?xf64>
    %44 = vector.transfer_read %43[%c0], %d0: memref<?xf64>, vector<64xf64>
    vector.print %44 : vector<64xf64>

    //
    // Inspect storage scheme of BlockCol.
    //
    // pointers(0)
    // indices(0)
    // values
    //
    // CHECK: ( 0, 7 )
    // CHECK: ( 0, 1, 2, 3, 4, 6, 7 )
    // CHECK: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 13, 0, 0, 2, 0, 4, 0,
    // CHECK-SAME: 0, 8, 10, 14, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
    // CHECK-SAME: 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 15, 0, 17, 3, 0, 0, 0, 0, 9, 12, 16, 0, 0 )
    //
    %45 = sparse_tensor.pointers %y { dimension = 0 : index } : tensor<10x8xf64, #BlockCol> to memref<?xindex>
    %46 = vector.transfer_read %45[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %46 : vector<2xindex>
    %47 = sparse_tensor.indices %y { dimension = 0 : index } : tensor<10x8xf64, #BlockCol> to memref<?xindex>
    %48 = vector.transfer_read %47[%c0], %c0: memref<?xindex>, vector<7xindex>
    vector.print %48 : vector<7xindex>
    %49 = sparse_tensor.values %y : tensor<10x8xf64, #BlockCol> to memref<?xf64>
    %50 = vector.transfer_read %49[%c0], %d0: memref<?xf64>, vector<70xf64>
    vector.print %50 : vector<70xf64>

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<10x8xf64, #Dense>
    bufferization.dealloc_tensor %1 : tensor<10x8xf64, #CSR>
    bufferization.dealloc_tensor %2 : tensor<10x8xf64, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<10x8xf64, #CSC>
    bufferization.dealloc_tensor %4 : tensor<10x8xf64, #DCSC>
    bufferization.dealloc_tensor %x : tensor<10x8xf64, #BlockRow>
    bufferization.dealloc_tensor %y : tensor<10x8xf64, #BlockCol>

    return
  }
}
