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
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa))
}>

#SortedCOOPermuted = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed(nonunique), d0 : singleton(soa)),
}>

#SortedCOO3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, soa), d2 : singleton(soa))
}>

#SortedCOO3DPermuted = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed(nonunique), d0 : singleton(nonunique, soa), d1 : singleton(soa))

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

  func.func @main() {
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
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 4, 256 )
    // CHECK-NEXT: lvl = ( 4, 256 )
    // CHECK-NEXT: pos[0] : ( 0, 17 )
    // CHECK-NEXT: crd[0] : ( 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 126, 127, 254, 1, 253, 2, 0, 1, 3, 98, 126, 127, 128, 249, 253, 255 )
    // CHECK-NEXT: values : ( -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?x?xf64, #SortedCOO>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 4, 256 )
    // CHECK-NEXT: lvl = ( 256, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 17 )
    // CHECK-NEXT: crd[0] : ( 0, 0, 1, 1, 2, 3, 98, 126, 126, 127, 127, 128, 249, 253, 253, 254, 255 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 1, 3, 2, 3, 3, 0, 3, 0, 3, 3, 3, 1, 3, 0, 3 )
    // CHECK-NEXT: values : ( -1, 8, -5, -9, -7, 10, -11, 2, 12, -3, -13, 14, -15, 6, 16, 4, -17 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<?x?xf64, #SortedCOOPermuted>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 17 )
    // CHECK-NEXT: crd[0] : ( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
    // CHECK-NEXT: crd[1] : ( 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2 )
    // CHECK-NEXT: crd[2] : ( 2, 3, 1, 2, 0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3, 1, 2 )
    // CHECK-NEXT: values : ( 3, 63, 11, 100, 66, 61, 13, 43, 77, 10, 46, 61, 53, 3, 75, 22, 18 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %2 : tensor<?x?x?xf64, #SortedCOO3D>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 4, 2, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 17 )
    // CHECK-NEXT: crd[0] : ( 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1 )
    // CHECK-NEXT: crd[2] : ( 2, 0, 1, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 1 )
    // CHECK-NEXT: values : ( 66, 77, 61, 11, 61, 53, 22, 3, 100, 13, 10, 3, 18, 63, 43, 46, 75 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %3 : tensor<?x?x?xf64, #SortedCOO3DPermuted>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 5, 4 )
    // CHECK-NEXT: lvl = ( 5, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 2, 3, 4 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 0, 3, 1, 1 )
    // CHECK-NEXT: values : ( 6, 5, 4, 3, 2, 11 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %4 : tensor<?x?xf64, #SortedCOO>

    // And last but not least, an actual operation applied to COO.
    // Note that this performs the operation "in place".
    %5 = call @sparse_scale(%4) : (tensor<?x?xf64, #SortedCOO>) -> tensor<?x?xf64, #SortedCOO>

    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 5, 4 )
    // CHECK-NEXT: lvl = ( 5, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 2, 3, 4 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 0, 3, 1, 1 )
    // CHECK-NEXT: values : ( 12, 10, 8, 6, 4, 22 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %5 : tensor<?x?xf64, #SortedCOO>

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<?x?xf64, #SortedCOO>
    bufferization.dealloc_tensor %1 : tensor<?x?xf64, #SortedCOOPermuted>
    bufferization.dealloc_tensor %2 : tensor<?x?x?xf64, #SortedCOO3D>
    bufferization.dealloc_tensor %3 : tensor<?x?x?xf64, #SortedCOO3DPermuted>
    bufferization.dealloc_tensor %4 : tensor<?x?xf64, #SortedCOO>

    return
  }
}
