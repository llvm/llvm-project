// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// RUN: TENSOR1="%mlir_src_dir/test/Integration/data/mttkrp_b.tns" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = !llvm.ptr<i8>

#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

#SortedCOOPermuted = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

#SortedCOO3D = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ]
}>

#SortedCOO3DPermuted = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = X(i,j) * 2.0"
}

//
// Tests reading in matrix/tensor from file into Sorted COO formats
// as well as applying various operations to this format.
//
module {

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // A kernel that scales a sparse matrix A by a factor of 2.0.
  //
  func.func @sparse_scale(%argx: tensor<?x?xf64, #SortedCOO>)
                              -> tensor<?x?xf64, #SortedCOO> {
    %c = arith.constant 2.0 : f64
    %0 = linalg.generic #trait_scale
      outs(%argx: tensor<?x?xf64, #SortedCOO>) {
        ^bb(%x: f64):
          %1 = arith.mulf %x, %c : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #SortedCOO>
    return %0 : tensor<?x?xf64, #SortedCOO>
  }

  func.func @dumpi(%arg0: memref<?xindex>) {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %arg0[%c0], %c0: memref<?xindex>, vector<20xindex>
    vector.print %v : vector<20xindex>
    return
  }

  func.func @dumpf(%arg0: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %nan = arith.constant 0x7FF0000001000000 : f64
    %v = vector.transfer_read %arg0[%c0], %nan: memref<?xf64>, vector<20xf64>
    vector.print %v : vector<20xf64>
    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName0 = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %fileName1 = call @getTensorFilename(%c1) : (index) -> (!Filename)

    // Read the sparse tensors from file, construct sparse storage.
    %0 = sparse_tensor.new %fileName0 : !Filename to tensor<?x?xf64, #SortedCOO>
    %1 = sparse_tensor.new %fileName0 : !Filename to tensor<?x?xf64, #SortedCOOPermuted>
    %2 = sparse_tensor.new %fileName1 : !Filename to tensor<?x?x?xf64, #SortedCOO3D>
    %3 = sparse_tensor.new %fileName1 : !Filename to tensor<?x?x?xf64, #SortedCOO3DPermuted>

    // Conversion from literal.
    %m = arith.constant sparse<
       [ [0,0], [1,3], [2,0], [2,3], [3,1], [4,1] ],
         [6.0, 5.0, 4.0, 3.0, 2.0, 11.0 ]
    > : tensor<5x4xf64>
    %4 = sparse_tensor.convert %m : tensor<5x4xf64> to tensor<?x?xf64, #SortedCOO>

    //
    // CHECK:      ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 126, 127, 254, 1, 253, 2, 0, 1, 3, 98, 126, 127, 128, 249, 253, 255, 0, 0, 0 )
    // CHECK-NEXT: ( -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, nan, nan, nan )
    //
    %p0 = sparse_tensor.pointers %0 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i00 = sparse_tensor.indices %0 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i01 = sparse_tensor.indices %0 { dimension = 1 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %v0 = sparse_tensor.values %0
      : tensor<?x?xf64, #SortedCOO> to memref<?xf64>
    call @dumpi(%p0)  : (memref<?xindex>) -> ()
    call @dumpi(%i00) : (memref<?xindex>) -> ()
    call @dumpi(%i01) : (memref<?xindex>) -> ()
    call @dumpf(%v0)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 1, 1, 2, 3, 98, 126, 126, 127, 127, 128, 249, 253, 253, 254, 255, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 1, 3, 2, 3, 3, 0, 3, 0, 3, 3, 3, 1, 3, 0, 3, 0, 0, 0 )
    // CHECK-NEXT: ( -1, 8, -5, -9, -7, 10, -11, 2, 12, -3, -13, 14, -15, 6, 16, 4, -17, nan, nan, nan )
    //
    %p1 = sparse_tensor.pointers %1 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xindex>
    %i10 = sparse_tensor.indices %1 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xindex>
    %i11 = sparse_tensor.indices %1 { dimension = 1 : index }
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xindex>
    %v1 = sparse_tensor.values %1
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xf64>
    call @dumpi(%p1)  : (memref<?xindex>) -> ()
    call @dumpi(%i10) : (memref<?xindex>) -> ()
    call @dumpi(%i11) : (memref<?xindex>) -> ()
    call @dumpf(%v1)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0 )
    // CHECK-NEXT: ( 3, 63, 11, 100, 66, 61, 13, 43, 77, 10, 46, 61, 53, 3, 75, 22, 18, nan, nan, nan )
    //
    %p2 = sparse_tensor.pointers %2 { dimension = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex>
    %i20 = sparse_tensor.indices %2 { dimension = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex>
    %i21 = sparse_tensor.indices %2 { dimension = 1 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex>
    %i22 = sparse_tensor.indices %2 { dimension = 2 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex>
    %v2 = sparse_tensor.values %2
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xf64>
    call @dumpi(%p2)  : (memref<?xindex>) -> ()
    call @dumpi(%i20) : (memref<?xindex>) -> ()
    call @dumpi(%i21) : (memref<?xindex>) -> ()
    call @dumpi(%i21) : (memref<?xindex>) -> ()
    call @dumpf(%v2)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0 )
    // CHECK-NEXT: ( 66, 77, 61, 11, 61, 53, 22, 3, 100, 13, 10, 3, 18, 63, 43, 46, 75, nan, nan, nan )
    //
    %p3 = sparse_tensor.pointers %3 { dimension = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex>
    %i30 = sparse_tensor.indices %3 { dimension = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex>
    %i31 = sparse_tensor.indices %3 { dimension = 1 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex>
    %i32 = sparse_tensor.indices %3 { dimension = 2 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex>
    %v3 = sparse_tensor.values %3
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xf64>
    call @dumpi(%p3)  : (memref<?xindex>) -> ()
    call @dumpi(%i30) : (memref<?xindex>) -> ()
    call @dumpi(%i31) : (memref<?xindex>) -> ()
    call @dumpi(%i31) : (memref<?xindex>) -> ()
    call @dumpf(%v3)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 6, 5, 4, 3, 2, 11, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan )
    //
    %p4 = sparse_tensor.pointers %4 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i40 = sparse_tensor.indices %4 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i41 = sparse_tensor.indices %4 { dimension = 1 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %v4 = sparse_tensor.values %4
      : tensor<?x?xf64, #SortedCOO> to memref<?xf64>
    call @dumpi(%p4)  : (memref<?xindex>) -> ()
    call @dumpi(%i40) : (memref<?xindex>) -> ()
    call @dumpi(%i41) : (memref<?xindex>) -> ()
    call @dumpf(%v4)  : (memref<?xf64>) -> ()

    // And last but not least, an actual operation applied to COO.
    // Note that this performs the operation "in place".
    %5 = call @sparse_scale(%4) : (tensor<?x?xf64, #SortedCOO>) -> tensor<?x?xf64, #SortedCOO>

    //
    // CHECK-NEXT: ( 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 12, 10, 8, 6, 4, 22, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan )
    //
    %p5 = sparse_tensor.pointers %5 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i50 = sparse_tensor.indices %5 { dimension = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i51 = sparse_tensor.indices %5 { dimension = 1 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %v5 = sparse_tensor.values %5
      : tensor<?x?xf64, #SortedCOO> to memref<?xf64>
    call @dumpi(%p5)  : (memref<?xindex>) -> ()
    call @dumpi(%i50) : (memref<?xindex>) -> ()
    call @dumpi(%i51) : (memref<?xindex>) -> ()
    call @dumpf(%v5)  : (memref<?xf64>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<?x?xf64, #SortedCOO>
    bufferization.dealloc_tensor %1 : tensor<?x?xf64, #SortedCOOPermuted>
    bufferization.dealloc_tensor %2 : tensor<?x?x?xf64, #SortedCOO3D>
    bufferization.dealloc_tensor %3 : tensor<?x?x?xf64, #SortedCOO3DPermuted>
    bufferization.dealloc_tensor %4 : tensor<?x?xf64, #SortedCOO>

    return
  }
}
