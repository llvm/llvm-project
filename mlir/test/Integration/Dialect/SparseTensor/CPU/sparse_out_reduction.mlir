// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false  enable-buffer-initialization=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#SparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ]
}>

#redsum = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>, // A
    affine_map<(i,j,k) -> (i,j,k)>, // B
    affine_map<(i,j,k) -> (i,j)>    // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) = SUM_k A(i,j,k) * B(i,j,k)"
}

module {
  func.func @redsum(%arga: tensor<?x?x?xi32, #SparseTensor>,
               %argb: tensor<?x?x?xi32, #SparseTensor>)
                   -> tensor<?x?xi32, #SparseMatrix> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xi32, #SparseTensor>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xi32, #SparseTensor>
    %xinit = bufferization.alloc_tensor(%d0, %d1): tensor<?x?xi32, #SparseMatrix>
    %0 = linalg.generic #redsum
      ins(%arga, %argb: tensor<?x?x?xi32, #SparseTensor>,
                        tensor<?x?x?xi32, #SparseTensor>)
      outs(%xinit: tensor<?x?xi32, #SparseMatrix>) {
        ^bb(%a: i32, %b: i32, %x: i32):
          %0 = arith.muli %a, %b : i32
          %1 = arith.addi %x, %0 : i32
          linalg.yield %1 : i32
    } -> tensor<?x?xi32, #SparseMatrix>
    return %0 : tensor<?x?xi32, #SparseMatrix>
  }

  // Driver method to call and verify tensor kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    // Setup very sparse 3-d tensors.
    %t1 = arith.constant sparse<
       [ [1,1,3], [2,0,0], [2,2,1], [2,2,2], [2,2,3] ], [ 1, 2, 3, 4, 5 ]
    > : tensor<3x3x4xi32>
    %t2 = arith.constant sparse<
       [ [1,0,0], [1,1,3], [2,2,1], [2,2,3] ], [ 6, 7, 8, 9 ]
    > : tensor<3x3x4xi32>
    %st1 = sparse_tensor.convert %t1
      : tensor<3x3x4xi32> to tensor<?x?x?xi32, #SparseTensor>
    %st2 = sparse_tensor.convert %t2
      : tensor<3x3x4xi32> to tensor<?x?x?xi32, #SparseTensor>

    // Call kernel.
    %0 = call @redsum(%st1, %st2)
      : (tensor<?x?x?xi32, #SparseTensor>,
         tensor<?x?x?xi32, #SparseTensor>) -> tensor<?x?xi32, #SparseMatrix>

    //
    // Verify results. Only two entries stored in result. Correct structure.
    //
    // CHECK: ( 7, 69, 0, 0 )
    // CHECK-NEXT: ( ( 0, 0, 0 ), ( 0, 7, 0 ), ( 0, 0, 69 ) )
    //
    %val = sparse_tensor.values %0
      : tensor<?x?xi32, #SparseMatrix> to memref<?xi32>
    %vv = vector.transfer_read %val[%c0], %i0: memref<?xi32>, vector<4xi32>
    vector.print %vv : vector<4xi32>
    %dm = sparse_tensor.convert %0
      : tensor<?x?xi32, #SparseMatrix> to tensor<?x?xi32>
    %vm = vector.transfer_read %dm[%c0, %c0], %i0: tensor<?x?xi32>, vector<3x3xi32>
    vector.print %vm : vector<3x3xi32>

    // Release the resources.
    bufferization.dealloc_tensor %st1 : tensor<?x?x?xi32, #SparseTensor>
    bufferization.dealloc_tensor %st2 : tensor<?x?x?xi32, #SparseTensor>
    bufferization.dealloc_tensor %0 : tensor<?x?xi32, #SparseMatrix>
    return
  }
}
