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
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// If SVE is available, do the same run, but now with direct IR generation and VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

#trait_1d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = a(i) op i"
}

#trait_2d = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) op i op j"
}

//
// Test with indices and sparse inputs. All outputs are dense.
//
module {

  //
  // Kernel that uses index in the index notation (conjunction).
  //
  func.func @sparse_index_1d_conj(%arga: tensor<8xi64, #SparseVector>,
                                  %out: tensor<8xi64>) -> tensor<8xi64> {
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%out: tensor<8xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.muli %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64>
    return %r : tensor<8xi64>
  }

  //
  // Kernel that uses index in the index notation (disjunction).
  //
  func.func @sparse_index_1d_disj(%arga: tensor<8xi64, #SparseVector>,
                                  %out: tensor<8xi64>) -> tensor<8xi64> {
    %r = linalg.generic #trait_1d
        ins(%arga: tensor<8xi64, #SparseVector>)
       outs(%out: tensor<8xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %ii = arith.index_cast %i : index to i64
          %m1 = arith.addi %a, %ii : i64
          linalg.yield %m1 : i64
    } -> tensor<8xi64>
    return %r : tensor<8xi64>
  }

  //
  // Kernel that uses indices in the index notation (conjunction).
  //
  func.func @sparse_index_2d_conj(%arga: tensor<3x4xi64, #SparseMatrix>,
                                  %out: tensor<3x4xi64>) -> tensor<3x4xi64> {
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%out: tensor<3x4xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.muli %ii, %a : i64
          %m2 = arith.muli %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64>
    return %r : tensor<3x4xi64>
  }

  //
  // Kernel that uses indices in the index notation (disjunction).
  //
  func.func @sparse_index_2d_disj(%arga: tensor<3x4xi64, #SparseMatrix>,
                                  %out: tensor<3x4xi64>) -> tensor<3x4xi64> {
    %r = linalg.generic #trait_2d
        ins(%arga: tensor<3x4xi64, #SparseMatrix>)
       outs(%out: tensor<3x4xi64>) {
        ^bb(%a: i64, %x: i64):
          %i = linalg.index 0 : index
          %j = linalg.index 1 : index
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %m1 = arith.addi %ii, %a : i64
          %m2 = arith.addi %jj, %m1 : i64
          linalg.yield %m2 : i64
    } -> tensor<3x4xi64>
    return %r : tensor<3x4xi64>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %du = arith.constant -1 : i64

    // Setup input sparse vector.
    %v1 = arith.constant sparse<[[2], [4]], [ 10, 20]> : tensor<8xi64>
    %sv = sparse_tensor.convert %v1 : tensor<8xi64> to tensor<8xi64, #SparseVector>

    // Setup input "sparse" vector.
    %v2 = arith.constant dense<[ 1,  2,  4,  8,  16,  32,  64,  128 ]> : tensor<8xi64>
    %dv = sparse_tensor.convert %v2 : tensor<8xi64> to tensor<8xi64, #SparseVector>

    // Setup input sparse matrix.
    %m1 = arith.constant sparse<[[1,1], [2,3]], [10, 20]> : tensor<3x4xi64>
    %sm = sparse_tensor.convert %m1 : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Setup input "sparse" matrix.
    %m2 = arith.constant dense <[ [ 1,  1,  1,  1 ],
                                  [ 1,  2,  1,  1 ],
                                  [ 1,  1,  3,  4 ] ]> : tensor<3x4xi64>
    %dm = sparse_tensor.convert %m2 : tensor<3x4xi64> to tensor<3x4xi64, #SparseMatrix>

    // Setup out tensors.
    %init_8 = bufferization.alloc_tensor() : tensor<8xi64>
    %init_3_4 = bufferization.alloc_tensor() : tensor<3x4xi64>

    // Call the kernels.
    %0 = call @sparse_index_1d_conj(%sv, %init_8) : (tensor<8xi64, #SparseVector>, tensor<8xi64>) -> tensor<8xi64>
    %1 = call @sparse_index_1d_disj(%sv, %init_8) : (tensor<8xi64, #SparseVector>, tensor<8xi64>) -> tensor<8xi64>
    %2 = call @sparse_index_1d_conj(%dv, %init_8) : (tensor<8xi64, #SparseVector>, tensor<8xi64>) -> tensor<8xi64>
    %3 = call @sparse_index_1d_disj(%dv, %init_8) : (tensor<8xi64, #SparseVector>, tensor<8xi64>) -> tensor<8xi64>
    %4 = call @sparse_index_2d_conj(%sm, %init_3_4) : (tensor<3x4xi64, #SparseMatrix>, tensor<3x4xi64>) -> tensor<3x4xi64>
    %5 = call @sparse_index_2d_disj(%sm, %init_3_4) : (tensor<3x4xi64, #SparseMatrix>, tensor<3x4xi64>) -> tensor<3x4xi64>
    %6 = call @sparse_index_2d_conj(%dm, %init_3_4) : (tensor<3x4xi64, #SparseMatrix>, tensor<3x4xi64>) -> tensor<3x4xi64>
    %7 = call @sparse_index_2d_disj(%dm, %init_3_4) : (tensor<3x4xi64, #SparseMatrix>, tensor<3x4xi64>) -> tensor<3x4xi64>

    //
    // Verify result.
    //
    // CHECK:      ( 0, 0, 20, 0, 80, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 12, 3, 24, 5, 6, 7 )
    // CHECK-NEXT: ( 0, 2, 8, 24, 64, 160, 384, 896 )
    // CHECK-NEXT: ( 1, 3, 6, 11, 20, 37, 70, 135 )
    // CHECK-NEXT: ( ( 0, 0, 0, 0 ), ( 0, 10, 0, 0 ), ( 0, 0, 0, 120 ) )
    // CHECK-NEXT: ( ( 0, 1, 2, 3 ), ( 1, 12, 3, 4 ), ( 2, 3, 4, 25 ) )
    // CHECK-NEXT: ( ( 0, 0, 0, 0 ), ( 0, 2, 2, 3 ), ( 0, 2, 12, 24 ) )
    // CHECK-NEXT: ( ( 1, 2, 3, 4 ), ( 2, 4, 4, 5 ), ( 3, 4, 7, 9 ) )
    //
    %vv0 = vector.transfer_read %0[%c0], %du: tensor<8xi64>, vector<8xi64>
    %vv1 = vector.transfer_read %1[%c0], %du: tensor<8xi64>, vector<8xi64>
    %vv2 = vector.transfer_read %2[%c0], %du: tensor<8xi64>, vector<8xi64>
    %vv3 = vector.transfer_read %3[%c0], %du: tensor<8xi64>, vector<8xi64>
    %vv4 = vector.transfer_read %4[%c0,%c0], %du: tensor<3x4xi64>, vector<3x4xi64>
    %vv5 = vector.transfer_read %5[%c0,%c0], %du: tensor<3x4xi64>, vector<3x4xi64>
    %vv6 = vector.transfer_read %6[%c0,%c0], %du: tensor<3x4xi64>, vector<3x4xi64>
    %vv7 = vector.transfer_read %7[%c0,%c0], %du: tensor<3x4xi64>, vector<3x4xi64>
    vector.print %vv0 : vector<8xi64>
    vector.print %vv1 : vector<8xi64>
    vector.print %vv2 : vector<8xi64>
    vector.print %vv3 : vector<8xi64>
    vector.print %vv4 : vector<3x4xi64>
    vector.print %vv5 : vector<3x4xi64>
    vector.print %vv6 : vector<3x4xi64>
    vector.print %vv7 : vector<3x4xi64>

    // Release resources.
    bufferization.dealloc_tensor %sv : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %dv : tensor<8xi64, #SparseVector>
    bufferization.dealloc_tensor %sm : tensor<3x4xi64, #SparseMatrix>
    bufferization.dealloc_tensor %dm : tensor<3x4xi64, #SparseMatrix>

    return
  }
}
