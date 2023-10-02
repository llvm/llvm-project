//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
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
    %init_8 = tensor.empty() : tensor<8xi64>
    %init_3_4 = tensor.empty() : tensor<3x4xi64>

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
