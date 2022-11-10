// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#TensorCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense", "compressed" ]
}>

#TensorRow = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "dense" ]
}>

module {

  func.func @dump(%arg0: tensor<5x4x3xf64, #TensorCSR>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %i0 = sparse_tensor.indices  %arg0 { dimension = 0 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %p2 = sparse_tensor.pointers %arg0 { dimension = 2 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %i2 = sparse_tensor.indices  %arg0 { dimension = 2 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<5x4x3xf64, #TensorCSR> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vi0 : vector<2xindex>
    %vp2 = vector.transfer_read %p2[%c0], %c0: memref<?xindex>, vector<9xindex>
    vector.print %vp2 : vector<9xindex>
    %vi2 = vector.transfer_read %i2[%c0], %c0: memref<?xindex>, vector<5xindex>
    vector.print %vi2 : vector<5xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<5xf64>
    vector.print %vv : vector<5xf64>
    return
  }

  func.func @dump_row(%arg0: tensor<5x4x3xf64, #TensorRow>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.pointers %arg0 { dimension = 0 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %i0 = sparse_tensor.indices  %arg0 { dimension = 0 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %p1 = sparse_tensor.pointers %arg0 { dimension = 1 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %i1 = sparse_tensor.indices  %arg0 { dimension = 1 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<5x4x3xf64, #TensorRow> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vi0 : vector<2xindex>
    %vp1 = vector.transfer_read %p1[%c0], %c0: memref<?xindex>, vector<3xindex>
    vector.print %vp1 : vector<3xindex>
    %vi1 = vector.transfer_read %i1[%c0], %c0: memref<?xindex>, vector<4xindex>
    vector.print %vi1 : vector<4xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<12xf64>
    vector.print %vv : vector<12xf64>
    return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %f1 = arith.constant 1.1 : f64
    %f2 = arith.constant 2.2 : f64
    %f3 = arith.constant 3.3 : f64
    %f4 = arith.constant 4.4 : f64
    %f5 = arith.constant 5.5 : f64

    //
    // CHECK:      ( 0, 2 )
    // CHECK-NEXT: ( 3, 4 )
    // CHECK-NEXT: ( 0, 2, 2, 2, 3, 3, 3, 4, 5 )
    // CHECK-NEXT: ( 1, 2, 1, 2, 2 )
    // CHECK-NEXT: ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    //
    %tensora = bufferization.alloc_tensor() : tensor<5x4x3xf64, #TensorCSR>
    %tensor1 = sparse_tensor.insert %f1 into %tensora[%c3, %c0, %c1] : tensor<5x4x3xf64, #TensorCSR>
    %tensor2 = sparse_tensor.insert %f2 into %tensor1[%c3, %c0, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensor3 = sparse_tensor.insert %f3 into %tensor2[%c3, %c3, %c1] : tensor<5x4x3xf64, #TensorCSR>
    %tensor4 = sparse_tensor.insert %f4 into %tensor3[%c4, %c2, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensor5 = sparse_tensor.insert %f5 into %tensor4[%c4, %c3, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensorm = sparse_tensor.load %tensor5 hasInserts : tensor<5x4x3xf64, #TensorCSR>
    call @dump(%tensorm) : (tensor<5x4x3xf64, #TensorCSR>) -> ()

    //
    // CHECK-NEXT: ( 0, 2 )
    // CHECK-NEXT: ( 3, 4 )
    // CHECK-NEXT: ( 0, 2, 4 )
    // CHECK-NEXT: ( 0, 3, 2, 3 )
    // CHECK-NEXT: ( 0, 1.1, 2.2, 0, 3.3, 0, 0, 0, 4.4, 0, 0, 5.5 )
    //
    %rowa = bufferization.alloc_tensor() : tensor<5x4x3xf64, #TensorRow>
    %row1 = sparse_tensor.insert %f1 into %rowa[%c3, %c0, %c1] : tensor<5x4x3xf64, #TensorRow>
    %row2 = sparse_tensor.insert %f2 into %row1[%c3, %c0, %c2] : tensor<5x4x3xf64, #TensorRow>
    %row3 = sparse_tensor.insert %f3 into %row2[%c3, %c3, %c1] : tensor<5x4x3xf64, #TensorRow>
    %row4 = sparse_tensor.insert %f4 into %row3[%c4, %c2, %c2] : tensor<5x4x3xf64, #TensorRow>
    %row5 = sparse_tensor.insert %f5 into %row4[%c4, %c3, %c2] : tensor<5x4x3xf64, #TensorRow>
    %rowm = sparse_tensor.load %row5 hasInserts : tensor<5x4x3xf64, #TensorRow>
    call @dump_row(%rowm) : (tensor<5x4x3xf64, #TensorRow>) -> ()

    // NOE sanity check.
    //
    // CHECK-NEXT: 5
    // CHECK-NEXT: 12
    //
    %noe1 = sparse_tensor.number_of_entries %tensorm : tensor<5x4x3xf64, #TensorCSR>
    vector.print %noe1 : index
    %noe2 = sparse_tensor.number_of_entries %rowm : tensor<5x4x3xf64, #TensorRow>
    vector.print %noe2 : index

    // Release resources.
    bufferization.dealloc_tensor %tensorm : tensor<5x4x3xf64, #TensorCSR>
    bufferization.dealloc_tensor %rowm : tensor<5x4x3xf64, #TensorRow>

    return
  }
}
