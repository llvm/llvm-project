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
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=4 reassociate-fp-reductions=true enable-index-optimizations=true enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>
#DenseVector = #sparse_tensor.encoding<{dimLevelType = ["dense"]}>

//
// Traits for 1-d tensor (aka vector) operations.
//
#trait_scale = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * 2.0"
}
#trait_scale_inpl = {
  indexing_maps = [
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) *= 2.0"
}
#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}
#trait_dot = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> ()>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) += a(i) * b(i)"
}

module {
  // Scales a sparse vector into a new sparse vector.
  func.func @vector_scale(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %s = arith.constant 2.0 : f64
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_scale
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %x: f64):
          %1 = arith.mulf %a, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Scales a sparse vector in place.
  func.func @vector_scale_inplace(%argx: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %s = arith.constant 2.0 : f64
    %0 = linalg.generic #trait_scale_inpl
      outs(%argx: tensor<?xf64, #SparseVector>) {
        ^bb(%x: f64):
          %1 = arith.mulf %x, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Adds two sparse vectors into a new sparse vector.
  func.func @vector_add(%arga: tensor<?xf64, #SparseVector>,
                   %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.addf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Multiplies two sparse vectors into a new sparse vector.
  func.func @vector_mul(%arga: tensor<?xf64, #SparseVector>,
                   %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Multiplies two sparse vectors into a new "annotated" dense vector.
  func.func @vector_mul_d(%arga: tensor<?xf64, #SparseVector>,
                     %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #DenseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #DenseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #DenseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #DenseVector>
    return %0 : tensor<?xf64, #DenseVector>
  }

  // Sum reduces dot product of two sparse vectors.
  func.func @vector_dotprod(%arga: tensor<?xf64, #SparseVector>,
                       %argb: tensor<?xf64, #SparseVector>,
                       %argx: tensor<f64>) -> tensor<f64> {
    %0 = linalg.generic #trait_dot
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%argx: tensor<f64>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          %2 = arith.addf %x, %1 : f64
          linalg.yield %2 : f64
    } -> tensor<f64>
    return %0 : tensor<f64>
  }

  // Dumps a sparse vector.
  func.func @dump(%arg0: tensor<?xf64, #SparseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xf64>, vector<16xf64>
    vector.print %1 : vector<16xf64>
    // Dump the dense vector to verify structure is correct.
    %dv = sparse_tensor.convert %arg0 : tensor<?xf64, #SparseVector> to tensor<?xf64>
    %2 = vector.transfer_read %dv[%c0], %d0: tensor<?xf64>, vector<32xf64>
    vector.print %2 : vector<32xf64>
    return
  }

  // Driver method to call and verify vector kernels.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant 1.1 : f64

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xf64>
    %v2 = arith.constant sparse<
       [ [1], [3], [4], [10], [16], [18], [21], [28], [29], [31] ],
         [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 ]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    // TODO: Use %sv1 when copying sparse tensors is supported.
    %sv1_dup = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xf64> to tensor<?xf64, #SparseVector>

    // Setup memory for a single reduction scalar.
    %x = tensor.from_elements %d1 : tensor<f64>

    // Call sparse vector kernels.
    %0 = call @vector_scale(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %1 = call @vector_scale_inplace(%sv1_dup)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %2 = call @vector_add(%1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %3 = call @vector_mul(%1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %4 = call @vector_mul_d(%1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #DenseVector>
    %5 = call @vector_dotprod(%1, %sv2, %x)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>, tensor<f64>) -> tensor<f64>

    //
    // Verify the results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    // CHECK-NEXT: ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 11, 0, 12, 13, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 15, 0, 16, 0, 0, 17, 0, 0, 0, 0, 0, 0, 18, 19, 0, 20 )
    // CHECK-NEXT: ( 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 0, 10, 12, 0, 0, 0, 0, 0, 0, 14, 16, 0, 18 )
    // CHECK-NEXT: ( 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 0, 10, 12, 0, 0, 0, 0, 0, 0, 14, 16, 0, 18 )
    // CHECK-NEXT: ( 2, 11, 16, 13, 14, 6, 15, 8, 16, 10, 29, 32, 35, 38, 0, 0 )
    // CHECK-NEXT: ( 2, 11, 0, 16, 13, 0, 0, 0, 0, 0, 14, 6, 0, 0, 0, 0, 15, 8, 16, 0, 10, 29, 0, 0, 0, 0, 0, 0, 32, 35, 0, 38 )
    // CHECK-NEXT: ( 48, 204, 252, 304, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 204, 0, 0, 0, 0, 0, 0, 252, 304, 0, 360 )
    // CHECK-NEXT: ( 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 204, 0, 0, 0, 0, 0, 0, 252, 304, 0, 360 )
    // CHECK-NEXT: 1169.1
    //

    call @dump(%sv1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump(%sv2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump(%0) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump(%1) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump(%2) : (tensor<?xf64, #SparseVector>) -> ()
    call @dump(%3) : (tensor<?xf64, #SparseVector>) -> ()
    %m4 = sparse_tensor.values %4 : tensor<?xf64, #DenseVector> to memref<?xf64>
    %v4 = vector.load %m4[%c0]: memref<?xf64>, vector<32xf64>
    vector.print %v4 : vector<32xf64>
    %v5 = tensor.extract %5[] : tensor<f64>
    vector.print %v5 : f64

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv1_dup : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %4 : tensor<?xf64, #DenseVector>
    return
  }
}
