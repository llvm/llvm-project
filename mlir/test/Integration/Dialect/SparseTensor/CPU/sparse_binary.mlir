// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>
#DCSR = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>

//
// Traits for tensor operations.
//
#trait_vec_scale = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}
#trait_vec_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}
#trait_mat_op = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>,  // B (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) OP B(i,j)"
}

//
// Contains test cases for the sparse_tensor.binary operator (different cases when left/right/overlap
// is empty/identity, etc).
//

module {
  // Creates a new sparse vector using the minimum values from two input sparse vectors.
  // When there is no overlap, include the present value in the output.
  func.func @vector_min(%arga: tensor<?xi32, #SparseVector>,
                        %argb: tensor<?xi32, #SparseVector>) -> tensor<?xi32, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xi32, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xi32, #SparseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xi32, #SparseVector>, tensor<?xi32, #SparseVector>)
        outs(%xv: tensor<?xi32, #SparseVector>) {
        ^bb(%a: i32, %b: i32, %x: i32):
          %1 = sparse_tensor.binary %a, %b : i32, i32 to i32
            overlap={
              ^bb0(%a0: i32, %b0: i32):
                %2 = arith.minsi %a0, %b0: i32
                sparse_tensor.yield %2 : i32
            }
            left=identity
            right=identity
          linalg.yield %1 : i32
    } -> tensor<?xi32, #SparseVector>
    return %0 : tensor<?xi32, #SparseVector>
  }

  // Creates a new sparse vector by multiplying a sparse vector with a dense vector.
  // When there is no overlap, leave the result empty.
  func.func @vector_mul(%arga: tensor<?xf64, #SparseVector>,
                        %argb: tensor<?xf64>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%a0: f64, %b0: f64):
                %ret = arith.mulf %a0, %b0 : f64
                sparse_tensor.yield %ret : f64
            }
            left={}
            right={}
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Take a set difference of two sparse vectors. The result will include only those
  // sparse elements present in the first, but not the second vector.
  func.func @vector_setdiff(%arga: tensor<?xf64, #SparseVector>,
                            %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={}
            left=identity
            right={}
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Return the index of each entry
  func.func @vector_index(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xi32, #SparseVector>
    %0 = linalg.generic #trait_vec_scale
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xi32, #SparseVector>) {
        ^bb(%a: f64, %x: i32):
          %idx = linalg.index 0 : index
          %1 = sparse_tensor.binary %a, %idx : f64, index to i32
            overlap={
              ^bb0(%x0: f64, %i: index):
                %ret = arith.index_cast %i : index to i32
                sparse_tensor.yield %ret : i32
            }
            left={}
            right={}
          linalg.yield %1 : i32
    } -> tensor<?xi32, #SparseVector>
    return %0 : tensor<?xi32, #SparseVector>
  }

  // Adds two sparse matrices when they intersect. Where they don't intersect,
  // negate the 2nd argument's values; ignore 1st argument-only values.
  func.func @matrix_intersect(%arga: tensor<?x?xf64, #DCSR>,
                              %argb: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xv = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_mat_op
       ins(%arga, %argb: tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = sparse_tensor.binary %a, %b: f64, f64 to f64
            overlap={
              ^bb0(%x0: f64, %y0: f64):
                %ret = arith.addf %x0, %y0 : f64
                sparse_tensor.yield %ret : f64
            }
            left={}
            right={
              ^bb0(%x1: f64):
                %lret = arith.negf %x1 : f64
                sparse_tensor.yield %lret : f64
            }
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Tensor addition (use semi-ring binary operation).
  func.func @add_tensor_1(%A: tensor<4x4xf64, #DCSR>,
                          %B: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    %0 = linalg.generic #trait_mat_op
      ins(%A, %B: tensor<4x4xf64, #DCSR>,
                  tensor<4x4xf64, #DCSR>)
      outs(%C: tensor<4x4xf64, #DCSR>) {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                %ret = arith.addf %x, %y : f64
                sparse_tensor.yield %ret : f64
            }
            left=identity
            right=identity
          linalg.yield %result : f64
      } -> tensor<4x4xf64, #DCSR>
    return %0 : tensor<4x4xf64, #DCSR>
  }

  // Same as @add_tensor_1, but use sparse_tensor.yield instead of identity to yield value.
  func.func @add_tensor_2(%A: tensor<4x4xf64, #DCSR>,
                          %B: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    %0 = linalg.generic #trait_mat_op
      ins(%A, %B: tensor<4x4xf64, #DCSR>,
                  tensor<4x4xf64, #DCSR>)
      outs(%C: tensor<4x4xf64, #DCSR>) {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                %ret = arith.addf %x, %y : f64
                sparse_tensor.yield %ret : f64
            }
            left={
              ^bb0(%x: f64):
                sparse_tensor.yield %x : f64
            }
            right={
              ^bb0(%y: f64):
                sparse_tensor.yield %y : f64
            }
          linalg.yield %result : f64
      } -> tensor<4x4xf64, #DCSR>
    return %0 : tensor<4x4xf64, #DCSR>
  }

  // Performs triangular add/sub operation (using semi-ring binary op).
  func.func @triangular(%A: tensor<4x4xf64, #DCSR>,
                        %B: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    %0 = linalg.generic #trait_mat_op
      ins(%A, %B: tensor<4x4xf64, #DCSR>,
                  tensor<4x4xf64, #DCSR>)
      outs(%C: tensor<4x4xf64, #DCSR>) {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %upperTriangleResult = arith.addf %x, %y : f64
                %lowerTriangleResult = arith.subf %x, %y : f64
                %ret = arith.select %cmp, %upperTriangleResult, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
            left=identity
            right={
              ^bb0(%y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %lowerTriangleResult = arith.negf %y : f64
                %ret = arith.select %cmp, %y, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<4x4xf64, #DCSR>
    return %0 : tensor<4x4xf64, #DCSR>
  }

  // Perform sub operation (using semi-ring binary op) with a constant threshold.
  func.func @sub_with_thres(%A: tensor<4x4xf64, #DCSR>,
                            %B: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    // Defines out-block constant bounds.
    %thres_out_up = arith.constant 2.0 : f64
    %thres_out_lo = arith.constant -2.0 : f64

    %0 = linalg.generic #trait_mat_op
      ins(%A, %B: tensor<4x4xf64, #DCSR>,
                  tensor<4x4xf64, #DCSR>)
      outs(%C: tensor<4x4xf64, #DCSR>) {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                // Defines in-block constant bounds.
                %thres_up = arith.constant 1.0 : f64
                %thres_lo = arith.constant -1.0 : f64
                %result = arith.subf %x, %y : f64
                %cmp = arith.cmpf "oge", %result, %thres_up : f64
                %tmp = arith.select %cmp, %thres_up, %result : f64
                %cmp1 = arith.cmpf "ole", %tmp, %thres_lo : f64
                %ret = arith.select %cmp1, %thres_lo, %tmp : f64
                sparse_tensor.yield %ret : f64
            }
            left={
              ^bb0(%x: f64):
                // Uses out-block constant bounds.
                %cmp = arith.cmpf "oge", %x, %thres_out_up : f64
                %tmp = arith.select %cmp, %thres_out_up, %x : f64
                %cmp1 = arith.cmpf "ole", %tmp, %thres_out_lo : f64
                %ret = arith.select %cmp1, %thres_out_lo, %tmp : f64
                sparse_tensor.yield %ret : f64
            }
            right={
              ^bb0(%y: f64):
                %ny = arith.negf %y : f64
                %cmp = arith.cmpf "oge", %ny, %thres_out_up : f64
                %tmp = arith.select %cmp, %thres_out_up, %ny : f64
                %cmp1 = arith.cmpf "ole", %tmp, %thres_out_lo : f64
                %ret = arith.select %cmp1, %thres_out_lo, %tmp : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<4x4xf64, #DCSR>
    return %0 : tensor<4x4xf64, #DCSR>
  }

  // Performs isEqual only on intersecting elements.
  func.func @intersect_equal(%A: tensor<4x4xf64, #DCSR>,
                             %B: tensor<4x4xf64, #DCSR>) -> tensor<4x4xi8, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xi8, #DCSR>
    %0 = linalg.generic #trait_mat_op
      ins(%A, %B: tensor<4x4xf64, #DCSR>,
                  tensor<4x4xf64, #DCSR>)
      outs(%C: tensor<4x4xi8, #DCSR>) {
        ^bb0(%a: f64, %b: f64, %c: i8) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to i8
            overlap={
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpf "oeq", %x, %y : f64
                %ret = arith.extui %cmp : i1 to i8
                sparse_tensor.yield %ret : i8
            }
            left={}
            right={}
          linalg.yield %result : i8
      } -> tensor<4x4xi8, #DCSR>
    return %0 : tensor<4x4xi8, #DCSR>
  }

  // Keeps values on left, negate value on right, ignore value when overlapping.
  func.func @only_left_right(%A: tensor<4x4xf64, #DCSR>,
                             %B: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    %0 = linalg.generic #trait_mat_op
      ins(%A, %B: tensor<4x4xf64, #DCSR>,
                  tensor<4x4xf64, #DCSR>)
      outs(%C: tensor<4x4xf64, #DCSR>) {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={}
            left=identity
            right={
              ^bb0(%y: f64):
                %ret = arith.negf %y : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<4x4xf64, #DCSR>
    return %0 : tensor<4x4xf64, #DCSR>
  }

  //
  // Utility functions to dump the value of a tensor.
  //

  func.func @dump_vec(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %3 = vector.transfer_read %dv[%c0], %d0: tensor<?xf64>, vector<32xf64>
    vector.print %3 : vector<32xf64>
    return
  }

  func.func @dump_vec_i32(%arg0: tensor<?xi32, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0 : i32
    %0 = sparse_tensor.values %arg0 : tensor<?xi32, #SparseVector> to memref<?xi32>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xi32>, vector<24xi32>
    vector.print %1 : vector<24xi32>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xi32, #SparseVector> to tensor<?xi32>
    %3 = vector.transfer_read %dv[%c0], %d0: tensor<?xi32>, vector<32xi32>
    vector.print %3 : vector<32xi32>
    return
  }

  func.func @dump_mat(%arg0: tensor<?x?xf64, #DCSR>) {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64>
    %1 = vector.transfer_read %dm[%c0, %c0], %d0: tensor<?x?xf64>, vector<4x8xf64>
    vector.print %1 : vector<4x8xf64>
    return
  }

  func.func @dump_mat_4x4(%A: tensor<4x4xf64, #DCSR>) {
    %c0 = arith.constant 0 : index
    %du = arith.constant 0.0 : f64

    %c = sparse_tensor.convert %A : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %v = vector.transfer_read %c[%c0, %c0], %du: tensor<4x4xf64>, vector<4x4xf64>
    vector.print %v : vector<4x4xf64>

    %1 = sparse_tensor.values %A : tensor<4x4xf64, #DCSR> to memref<?xf64>
    %2 = vector.transfer_read %1[%c0], %du: memref<?xf64>, vector<16xf64>
    vector.print %2 : vector<16xf64>

    return
  }

  func.func @dump_mat_4x4_i8(%A: tensor<4x4xi8, #DCSR>) {
    %c0 = arith.constant 0 : index
    %du = arith.constant 0 : i8

    %c = sparse_tensor.convert %A : tensor<4x4xi8, #DCSR> to tensor<4x4xi8>
    %v = vector.transfer_read %c[%c0, %c0], %du: tensor<4x4xi8>, vector<4x4xi8>
    vector.print %v : vector<4x4xi8>

    %1 = sparse_tensor.values %A : tensor<4x4xi8, #DCSR> to memref<?xi8>
    %2 = vector.transfer_read %1[%c0], %du: memref<?xi8>, vector<16xi8>
    vector.print %2 : vector<16xi8>

    return
  }

  // Driver method to call and verify kernels.
  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xf64>
    %v2 = arith.constant sparse<
       [ [1], [3], [4], [10], [16], [18], [21], [28], [29], [31] ],
         [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 ]
    > : tensor<32xf64>
    %v3 = arith.constant dense<
      [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
       0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1.]
    > : tensor<32xf64>
    %v1_si = arith.fptosi %v1 : tensor<32xf64> to tensor<32xi32>
    %v2_si = arith.fptosi %v2 : tensor<32xf64> to tensor<32xi32>

    %sv1 = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %sv1_si = sparse_tensor.convert %v1_si : tensor<32xi32> to tensor<?xi32, #SparseVector>
    %sv2_si = sparse_tensor.convert %v2_si : tensor<32xi32> to tensor<?xi32, #SparseVector>
    %dv3 = tensor.cast %v3 : tensor<32xf64> to tensor<?xf64>

    // Setup sparse matrices.
    %m1 = arith.constant sparse<
       [ [0,0], [0,1], [1,7], [2,2], [2,4], [2,7], [3,0], [3,2], [3,3] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<4x8xf64>
    %m2 = arith.constant sparse<
       [ [0,0], [0,7], [1,0], [1,6], [2,1], [2,7] ],
         [6.0, 5.0, 4.0, 3.0, 2.0, 1.0 ]
    > : tensor<4x8xf64>
    %sm1 = sparse_tensor.convert %m1 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>
    %sm2 = sparse_tensor.convert %m2 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>

    %m3 = arith.constant dense<
      [ [ 1.0, 0.0, 3.0, 0.0],
        [ 0.0, 2.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 4.0],
        [ 3.0, 4.0, 0.0, 0.0] ]> : tensor<4x4xf64>
    %m4 = arith.constant dense<
      [ [ 1.0, 0.0, 1.0, 1.0],
        [ 0.0, 0.5, 0.0, 0.0],
        [ 1.0, 5.0, 2.0, 0.0],
        [ 2.0, 0.0, 0.0, 0.0] ]> : tensor<4x4xf64>

    %sm3 = sparse_tensor.convert %m3 : tensor<4x4xf64> to tensor<4x4xf64, #DCSR>
    %sm4 = sparse_tensor.convert %m4 : tensor<4x4xf64> to tensor<4x4xf64, #DCSR>

    // Call sparse vector kernels.
    %0 = call @vector_min(%sv1_si, %sv2_si)
       : (tensor<?xi32, #SparseVector>,
          tensor<?xi32, #SparseVector>) -> tensor<?xi32, #SparseVector>
    %1 = call @vector_mul(%sv1, %dv3)
      : (tensor<?xf64, #SparseVector>,
         tensor<?xf64>) -> tensor<?xf64, #SparseVector>
    %2 = call @vector_setdiff(%sv1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %3 = call @vector_index(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xi32, #SparseVector>

    // Call sparse matrix kernels.
    %5 = call @matrix_intersect(%sm1, %sm2)
      : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %6 = call @add_tensor_1(%sm3, %sm4)
      : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    %7 = call @add_tensor_2(%sm3, %sm4)
      : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    %8 = call @triangular(%sm3, %sm4)
      : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    %9 = call @sub_with_thres(%sm3, %sm4)
      : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    %10 = call @intersect_equal(%sm3, %sm4)
      : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xi8, #DCSR>
    %11 = call @only_left_right(%sm3, %sm4)
      : (tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 11, 0, 12, 13, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 15, 0, 16, 0, 0, 17, 0, 0, 0, 0, 0, 0, 18, 19, 0, 20 )
    // CHECK-NEXT: ( 1, 11, 2, 13, 14, 3, 15, 4, 16, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 11, 0, 2, 13, 0, 0, 0, 0, 0, 14, 3, 0, 0, 0, 0, 15, 4, 16, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 0, 6, 3, 28, 0, 6, 56, 72, 9, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 28, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 56, 72, 0, 9 )
    // CHECK-NEXT: ( 1, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 3, 11, 17, 20, 21, 28, 29, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 17, 0, 0, 20, 21, 0, 0, 0, 0, 0, 0, 28, 29, 0, 31 )
    // CHECK-NEXT: ( ( 7, 0, 0, 0, 0, 0, 0, -5 ), ( -4, 0, 0, 0, 0, 0, -3, 0 ), ( 0, -2, 0, 0, 0, 0, 0, 7 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ) )
    // CHECK-NEXT: ( ( 2, 0, 4, 1 ), ( 0, 2.5, 0, 0 ), ( 1, 5, 2, 4 ), ( 5, 4, 0, 0 ) )
    // CHECK-NEXT:   ( 2, 4, 1, 2.5, 1, 5, 2, 4, 5, 4, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 2, 0, 4, 1 ), ( 0, 2.5, 0, 0 ), ( 1, 5, 2, 4 ), ( 5, 4, 0, 0 ) )
    // CHECK-NEXT:   ( 2, 4, 1, 2.5, 1, 5, 2, 4, 5, 4, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 2, 0, 4, 1 ), ( 0, 2.5, 0, 0 ), ( -1, -5, 2, 4 ), ( 1, 4, 0, 0 ) )
    // CHECK-NEXT:   ( 2, 4, 1, 2.5, -1, -5, 2, 4, 1, 4, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 0, 0, 1, -1 ), ( 0, 1, 0, 0 ), ( -1, -2, -2, 2 ), ( 1, 2, 0, 0 ) )
    // CHECK-NEXT:   ( 0, 1, -1, 1, -1, -2, -2, 2, 1, 2, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 1, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) )
    // CHECK-NEXT:   ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( ( 0, 0, 0, -1 ), ( 0, 0, 0, 0 ), ( -1, -5, -2, 4 ), ( 0, 4, 0, 0 ) )
    // CHECK-NEXT:   ( -1, -1, -5, -2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    //
    call @dump_vec(%sv1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%sv2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec_i32(%0) : (tensor<?xi32, #SparseVector>) -> ()
    call @dump_vec(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec(%2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump_vec_i32(%3) : (tensor<?xi32, #SparseVector>) -> ()
    call @dump_mat(%5) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump_mat_4x4(%6) : (tensor<4x4xf64, #DCSR>) -> ()
    call @dump_mat_4x4(%7) : (tensor<4x4xf64, #DCSR>) -> ()
    call @dump_mat_4x4(%8) : (tensor<4x4xf64, #DCSR>) -> ()
    call @dump_mat_4x4(%9) : (tensor<4x4xf64, #DCSR>) -> ()
    call @dump_mat_4x4_i8(%10) : (tensor<4x4xi8, #DCSR>) -> ()
    call @dump_mat_4x4(%11) : (tensor<4x4xf64, #DCSR>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv1_si : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %sv2_si : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %sm2 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %sm3 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %sm4 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %5 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %6 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %7 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %8 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %9 : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %10 : tensor<4x4xi8, #DCSR>
    bufferization.dealloc_tensor %11 : tensor<4x4xf64, #DCSR>
    return
  }
}
