// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#Dense = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "dense"]
}>

#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#Row = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ]
}>

module {

  func.func @dump_dense(%arg0: tensor<4x3xf64, #Dense>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %v = sparse_tensor.values %arg0 : tensor<4x3xf64, #Dense> to memref<?xf64>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<12xf64>
    vector.print %vv : vector<12xf64>
    return
  }

  func.func @dump_coo(%arg0: tensor<4x3xf64, #SortedCOO>) {
    %c0 = arith.constant 0 : index
    %cu = arith.constant -1 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<4x3xf64, #SortedCOO> to memref<?xindex>
    %i0 = sparse_tensor.indices  %arg0 { dimension = 0 : index } : tensor<4x3xf64, #SortedCOO> to memref<?xindex>
    %i1 = sparse_tensor.indices  %arg0 { dimension = 1 : index } : tensor<4x3xf64, #SortedCOO> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<4x3xf64, #SortedCOO> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %cu: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %cu: memref<?xindex>, vector<4xindex>
    vector.print %vi0 : vector<4xindex>
    %vi1 = vector.transfer_read %i1[%c0], %cu: memref<?xindex>, vector<4xindex>
    vector.print %vi1 : vector<4xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<4xf64>
    vector.print %vv : vector<4xf64>
    return
  }

  func.func @dump_csr(%arg0: tensor<4x3xf64, #CSR>) {
    %c0 = arith.constant 0 : index
    %cu = arith.constant -1 : index
    %fu = arith.constant 99.0 : f64
    %p1 = sparse_tensor.pointers %arg0 { dimension = 1 : index } : tensor<4x3xf64, #CSR> to memref<?xindex>
    %i1 = sparse_tensor.indices  %arg0 { dimension = 1 : index } : tensor<4x3xf64, #CSR> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<4x3xf64, #CSR> to memref<?xf64>
    %vp1 = vector.transfer_read %p1[%c0], %cu: memref<?xindex>, vector<5xindex>
    vector.print %vp1 : vector<5xindex>
    %vi1 = vector.transfer_read %i1[%c0], %cu: memref<?xindex>, vector<4xindex>
    vector.print %vi1 : vector<4xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<4xf64>
    vector.print %vv : vector<4xf64>
    return
  }

  func.func @dump_dcsr(%arg0: tensor<4x3xf64, #DCSR>) {
    %c0 = arith.constant 0 : index
    %cu = arith.constant -1 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<4x3xf64, #DCSR> to memref<?xindex>
    %i0 = sparse_tensor.indices  %arg0 { dimension = 0 : index } : tensor<4x3xf64, #DCSR> to memref<?xindex>
    %p1 = sparse_tensor.pointers %arg0 { dimension = 1 : index } : tensor<4x3xf64, #DCSR> to memref<?xindex>
    %i1 = sparse_tensor.indices  %arg0 { dimension = 1 : index } : tensor<4x3xf64, #DCSR> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<4x3xf64, #DCSR> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %cu: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %cu: memref<?xindex>, vector<3xindex>
    vector.print %vi0 : vector<3xindex>
    %vp1 = vector.transfer_read %p1[%c0], %cu: memref<?xindex>, vector<4xindex>
    vector.print %vp1 : vector<4xindex>
    %vi1 = vector.transfer_read %i1[%c0], %cu: memref<?xindex>, vector<4xindex>
    vector.print %vi1 : vector<4xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<4xf64>
    vector.print %vv : vector<4xf64>
    return
  }

  func.func @dump_row(%arg0: tensor<4x3xf64, #Row>) {
    %c0 = arith.constant 0 : index
    %cu = arith.constant -1 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<4x3xf64, #Row> to memref<?xindex>
    %i0 = sparse_tensor.indices  %arg0 { dimension = 0 : index } : tensor<4x3xf64, #Row> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<4x3xf64, #Row> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %cu: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %cu: memref<?xindex>, vector<3xindex>
    vector.print %vi0 : vector<3xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<9xf64>
    vector.print %vv : vector<9xf64>
    return
  }

  //
  // Main driver. We test the contents of various sparse tensor
  // schemes when they are still empty and after a few insertions.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %f1 = arith.constant 1.0 : f64
    %f2 = arith.constant 2.0 : f64
    %f3 = arith.constant 3.0 : f64
    %f4 = arith.constant 4.0 : f64

    //
    // Dense case.
    //
    // CHECK: ( 1, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 4 )
    //
    %densea = bufferization.alloc_tensor() : tensor<4x3xf64, #Dense>
    %dense1 = sparse_tensor.insert %f1 into %densea[%c0, %c0] : tensor<4x3xf64, #Dense>
    %dense2 = sparse_tensor.insert %f2 into %dense1[%c2, %c2] : tensor<4x3xf64, #Dense>
    %dense3 = sparse_tensor.insert %f3 into %dense2[%c3, %c0] : tensor<4x3xf64, #Dense>
    %dense4 = sparse_tensor.insert %f4 into %dense3[%c3, %c2] : tensor<4x3xf64, #Dense>
    %densem = sparse_tensor.load %dense4 hasInserts : tensor<4x3xf64, #Dense>
    call @dump_dense(%densem) : (tensor<4x3xf64, #Dense>) -> ()

    //
    // COO case.
    //
    // CHECK-NEXT: ( 0, 4 )
    // CHECK-NEXT: ( 0, 2, 3, 3 )
    // CHECK-NEXT: ( 0, 2, 0, 2 )
    // CHECK-NEXT: ( 1, 2, 3, 4 )
    //
    %cooa = bufferization.alloc_tensor() : tensor<4x3xf64, #SortedCOO>
    %coo1 = sparse_tensor.insert %f1 into %cooa[%c0, %c0] : tensor<4x3xf64, #SortedCOO>
    %coo2 = sparse_tensor.insert %f2 into %coo1[%c2, %c2] : tensor<4x3xf64, #SortedCOO>
    %coo3 = sparse_tensor.insert %f3 into %coo2[%c3, %c0] : tensor<4x3xf64, #SortedCOO>
    %coo4 = sparse_tensor.insert %f4 into %coo3[%c3, %c2] : tensor<4x3xf64, #SortedCOO>
    %coom = sparse_tensor.load %coo4 hasInserts : tensor<4x3xf64, #SortedCOO>
    call @dump_coo(%coom) : (tensor<4x3xf64, #SortedCOO>) -> ()

    //
    // CSR case.
    //
    // CHECK-NEXT: ( 0, 1, 1, 2, 4 )
    // CHECK-NEXT: ( 0, 2, 0, 2 )
    // CHECK-NEXT: ( 1, 2, 3, 4 )
    //
    %csra = bufferization.alloc_tensor() : tensor<4x3xf64, #CSR>
    %csr1 = sparse_tensor.insert %f1 into %csra[%c0, %c0] : tensor<4x3xf64, #CSR>
    %csr2 = sparse_tensor.insert %f2 into %csr1[%c2, %c2] : tensor<4x3xf64, #CSR>
    %csr3 = sparse_tensor.insert %f3 into %csr2[%c3, %c0] : tensor<4x3xf64, #CSR>
    %csr4 = sparse_tensor.insert %f4 into %csr3[%c3, %c2] : tensor<4x3xf64, #CSR>
    %csrm = sparse_tensor.load %csr4 hasInserts : tensor<4x3xf64, #CSR>
    call @dump_csr(%csrm) : (tensor<4x3xf64, #CSR>) -> ()

    //
    // DCSR case.
    //
    // CHECK-NEXT: ( 0, 3 )
    // CHECK-NEXT: ( 0, 2, 3 )
    // CHECK-NEXT: ( 0, 1, 2, 4 )
    // CHECK-NEXT: ( 0, 2, 0, 2 )
    // CHECK-NEXT: ( 1, 2, 3, 4 )
    //
    %dcsra = bufferization.alloc_tensor() : tensor<4x3xf64, #DCSR>
    %dcsr1 = sparse_tensor.insert %f1 into %dcsra[%c0, %c0] : tensor<4x3xf64, #DCSR>
    %dcsr2 = sparse_tensor.insert %f2 into %dcsr1[%c2, %c2] : tensor<4x3xf64, #DCSR>
    %dcsr3 = sparse_tensor.insert %f3 into %dcsr2[%c3, %c0] : tensor<4x3xf64, #DCSR>
    %dcsr4 = sparse_tensor.insert %f4 into %dcsr3[%c3, %c2] : tensor<4x3xf64, #DCSR>
    %dcsrm = sparse_tensor.load %dcsr4 hasInserts : tensor<4x3xf64, #DCSR>
    call @dump_dcsr(%dcsrm) : (tensor<4x3xf64, #DCSR>) -> ()

    //
    // Row case.
    //
    // CHECK-NEXT: ( 0, 3 )
    // CHECK-NEXT: ( 0, 2, 3 )
    // CHECK-NEXT: ( 1, 0, 0, 0, 0, 2, 3, 0, 4 )
    //
    %rowa = bufferization.alloc_tensor() : tensor<4x3xf64, #Row>
    %row1 = sparse_tensor.insert %f1 into %rowa[%c0, %c0] : tensor<4x3xf64, #Row>
    %row2 = sparse_tensor.insert %f2 into %row1[%c2, %c2] : tensor<4x3xf64, #Row>
    %row3 = sparse_tensor.insert %f3 into %row2[%c3, %c0] : tensor<4x3xf64, #Row>
    %row4 = sparse_tensor.insert %f4 into %row3[%c3, %c2] : tensor<4x3xf64, #Row>
    %rowm = sparse_tensor.load %row4 hasInserts : tensor<4x3xf64, #Row>
    call @dump_row(%rowm) : (tensor<4x3xf64, #Row>) -> ()

    //
    // NOE sanity check.
    //
    // CHECK-NEXT: 12
    // CHECK-NEXT: 4
    // CHECK-NEXT: 4
    // CHECK-NEXT: 4
    // CHECK-NEXT: 9
    //
    %noe1 = sparse_tensor.number_of_entries %densem : tensor<4x3xf64, #Dense>
    %noe2 = sparse_tensor.number_of_entries %coom : tensor<4x3xf64, #SortedCOO>
    %noe3 = sparse_tensor.number_of_entries %csrm : tensor<4x3xf64, #CSR>
    %noe4 = sparse_tensor.number_of_entries %dcsrm : tensor<4x3xf64, #DCSR>
    %noe5 = sparse_tensor.number_of_entries %rowm : tensor<4x3xf64, #Row>
    vector.print %noe1 : index
    vector.print %noe2 : index
    vector.print %noe3 : index
    vector.print %noe4 : index
    vector.print %noe5 : index

    // Release resources.
    bufferization.dealloc_tensor %densem : tensor<4x3xf64, #Dense>
    bufferization.dealloc_tensor %coom : tensor<4x3xf64, #SortedCOO>
    bufferization.dealloc_tensor %csrm : tensor<4x3xf64, #CSR>
    bufferization.dealloc_tensor %dcsrm : tensor<4x3xf64, #DCSR>
    bufferization.dealloc_tensor %rowm : tensor<4x3xf64, #Row>

    return
  }
}
