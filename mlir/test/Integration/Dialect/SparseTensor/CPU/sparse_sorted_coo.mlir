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
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// REDEFINE: TENSOR1="%mlir_src_dir/test/Integration/data/mttkrp_b.tns"
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=4 enable-buffer-initialization=true
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | env %{env} %{run_sve} | FileCheck %s %}

!Filename = !llvm.ptr

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#SortedCOOPermuted = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed(nonunique), d0 : singleton),
}>

#SortedCOO3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton)
}>

#SortedCOO3DPermuted = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed(nonunique), d0 : singleton(nonunique), d1 : singleton)

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

  func.func @dumpsi(%arg0: memref<?xindex, strided<[?], offset: ?>>) {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %arg0[%c0], %c0: memref<?xindex, strided<[?], offset: ?>>, vector<20xindex>
    vector.print %v : vector<20xindex>
    return
  }

  func.func @dumpf(%arg0: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %nan = arith.constant 0x0 : f64
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
    // CHECK-NEXT: ( -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 0, 0, 0 )
    //
    %p0 = sparse_tensor.positions %0 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i00 = sparse_tensor.coordinates %0 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex, strided<[?], offset: ?>>
    %i01 = sparse_tensor.coordinates %0 { level = 1 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex, strided<[?], offset: ?>>
    %v0 = sparse_tensor.values %0
      : tensor<?x?xf64, #SortedCOO> to memref<?xf64>
    call @dumpi(%p0)  : (memref<?xindex>) -> ()
    call @dumpsi(%i00) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i01) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpf(%v0)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 1, 1, 2, 3, 98, 126, 126, 127, 127, 128, 249, 253, 253, 254, 255, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 1, 3, 2, 3, 3, 0, 3, 0, 3, 3, 3, 1, 3, 0, 3, 0, 0, 0 )
    // CHECK-NEXT: ( -1, 8, -5, -9, -7, 10, -11, 2, 12, -3, -13, 14, -15, 6, 16, 4, -17, 0, 0, 0 )
    //
    %p1 = sparse_tensor.positions %1 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xindex>
    %i10 = sparse_tensor.coordinates %1 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xindex, strided<[?], offset: ?>>
    %i11 = sparse_tensor.coordinates %1 { level = 1 : index }
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xindex, strided<[?], offset: ?>>
    %v1 = sparse_tensor.values %1
      : tensor<?x?xf64, #SortedCOOPermuted> to memref<?xf64>
    call @dumpi(%p1)  : (memref<?xindex>) -> ()
    call @dumpsi(%i10) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i11) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpf(%v1)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0 )
    // CHECK-NEXT: ( 3, 63, 11, 100, 66, 61, 13, 43, 77, 10, 46, 61, 53, 3, 75, 22, 18, 0, 0, 0 )
    //
    %p2 = sparse_tensor.positions %2 { level = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex>
    %i20 = sparse_tensor.coordinates %2 { level = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex, strided<[?], offset: ?>>
    %i21 = sparse_tensor.coordinates %2 { level = 1 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex, strided<[?], offset: ?>>
    %i22 = sparse_tensor.coordinates %2 { level = 2 : index }
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xindex, strided<[?], offset: ?>>
    %v2 = sparse_tensor.values %2
      : tensor<?x?x?xf64, #SortedCOO3D> to memref<?xf64>
    call @dumpi(%p2)  : (memref<?xindex>) -> ()
    call @dumpsi(%i20) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i21) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i21) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpf(%v2)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0 )
    // CHECK-NEXT: ( 66, 77, 61, 11, 61, 53, 22, 3, 100, 13, 10, 3, 18, 63, 43, 46, 75, 0, 0, 0 )
    //
    %p3 = sparse_tensor.positions %3 { level = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex>
    %i30 = sparse_tensor.coordinates %3 { level = 0 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex, strided<[?], offset: ?>>
    %i31 = sparse_tensor.coordinates %3 { level = 1 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex, strided<[?], offset: ?>>
    %i32 = sparse_tensor.coordinates %3 { level = 2 : index }
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xindex, strided<[?], offset: ?>>
    %v3 = sparse_tensor.values %3
      : tensor<?x?x?xf64, #SortedCOO3DPermuted> to memref<?xf64>
    call @dumpi(%p3)  : (memref<?xindex>) -> ()
    call @dumpsi(%i30) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i31) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i31) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpf(%v3)  : (memref<?xf64>) -> ()

    //
    // CHECK-NEXT: ( 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
   // CHECK-NEXT: ( 6, 5, 4, 3, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    //
    %p4 = sparse_tensor.positions %4 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i40 = sparse_tensor.coordinates %4 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex, strided<[?], offset: ?>>
    %i41 = sparse_tensor.coordinates %4 { level = 1 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex, strided<[?], offset: ?>>
    %v4 = sparse_tensor.values %4
      : tensor<?x?xf64, #SortedCOO> to memref<?xf64>
    call @dumpi(%p4)  : (memref<?xindex>) -> ()
    call @dumpsi(%i40) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i41) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpf(%v4)  : (memref<?xf64>) -> ()

    // And last but not least, an actual operation applied to COO.
    // Note that this performs the operation "in place".
    %5 = call @sparse_scale(%4) : (tensor<?x?xf64, #SortedCOO>) -> tensor<?x?xf64, #SortedCOO>

    //
    // CHECK-NEXT: ( 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 12, 10, 8, 6, 4, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    //
    %p5 = sparse_tensor.positions %5 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex>
    %i50 = sparse_tensor.coordinates %5 { level = 0 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex, strided<[?], offset: ?>>
    %i51 = sparse_tensor.coordinates %5 { level = 1 : index }
      : tensor<?x?xf64, #SortedCOO> to memref<?xindex, strided<[?], offset: ?>>
    %v5 = sparse_tensor.values %5
      : tensor<?x?xf64, #SortedCOO> to memref<?xf64>
    call @dumpi(%p5)  : (memref<?xindex>) -> ()
    call @dumpsi(%i50) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
    call @dumpsi(%i51) : (memref<?xindex, strided<[?], offset: ?>>) -> ()
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
