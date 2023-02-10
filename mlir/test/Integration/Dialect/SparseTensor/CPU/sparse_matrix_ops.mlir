// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
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

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext --dlopen=%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

//
// Traits for 2-d tensor (aka matrix) operations.
//
#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * 2.0"
}
#trait_scale_inpl = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) *= 2.0"
}
#trait_op = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (i,j)>,  // B (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) OP B(i,j)"
}

module {
  func.func private @printMemrefF64(%ptr : tensor<*xf64>)

  // Scales a sparse matrix into a new sparse matrix.
  func.func @matrix_scale(%arga: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %s = arith.constant 2.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xm = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_scale
       ins(%arga: tensor<?x?xf64, #DCSR>)
        outs(%xm: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %x: f64):
          %1 = arith.mulf %a, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Scales a sparse matrix in place.
  func.func @matrix_scale_inplace(%argx: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %s = arith.constant 2.0 : f64
    %0 = linalg.generic #trait_scale_inpl
      outs(%argx: tensor<?x?xf64, #DCSR>) {
        ^bb(%x: f64):
          %1 = arith.mulf %x, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Adds two sparse matrices element-wise into a new sparse matrix.
  func.func @matrix_add(%arga: tensor<?x?xf64, #DCSR>,
                        %argb: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xv = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.addf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Multiplies two sparse matrices element-wise into a new sparse matrix.
  func.func @matrix_mul(%arga: tensor<?x?xf64, #DCSR>,
                        %argb: tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSR>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?xf64, #DCSR>
    %xv = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf64, #DCSR>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>)
        outs(%xv: tensor<?x?xf64, #DCSR>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
  }

  // Dump a sparse matrix.
  func.func @dump(%arg0: tensor<?x?xf64, #DCSR>) {
    %dm = sparse_tensor.convert %arg0 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64>
    %u = tensor.cast %dm : tensor<?x?xf64> to tensor<*xf64>
    call @printMemrefF64(%u) : (tensor<*xf64>) -> ()
    return
  }

  // Driver method to call and verify matrix kernels.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant 1.1 : f64

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
    // TODO: Use %sm1 when we support sparse tensor copies.
    %sm1_dup = sparse_tensor.convert %m1 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>
    %sm2 = sparse_tensor.convert %m2 : tensor<4x8xf64> to tensor<?x?xf64, #DCSR>

    // Call sparse matrix kernels.
    %0 = call @matrix_scale(%sm1)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %1 = call @matrix_scale_inplace(%sm1_dup)
      : (tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %2 = call @matrix_add(%1, %sm2)
      : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>
    %3 = call @matrix_mul(%1, %sm2)
      : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64, #DCSR>) -> tensor<?x?xf64, #DCSR>

    //
    // Verify the results.
    //
    // CHECK:      {{\[}}[1,   2,   0,   0,   0,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   3],
    // CHECK-NEXT: [0,   0,   4,   0,   5,   0,   0,   6],
    // CHECK-NEXT: [7,   0,   8,   9,   0,   0,   0,   0]]
    // CHECK:      {{\[}}[6,   0,   0,   0,   0,   0,   0,   5],
    // CHECK-NEXT: [4,   0,   0,   0,   0,   0,   3,   0],
    // CHECK-NEXT: [0,   2,   0,   0,   0,   0,   0,   1],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   0]]
    // CHECK:      {{\[}}[2,   4,   0,   0,   0,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   6],
    // CHECK-NEXT: [0,   0,   8,   0,   10,   0,   0,   12],
    // CHECK-NEXT: [14,   0,   16,   18,   0,   0,   0,   0]]
    // CHECK:      {{\[}}[2,   4,   0,   0,   0,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   6],
    // CHECK-NEXT: [0,   0,   8,   0,   10,   0,   0,   12],
    // CHECK-NEXT: [14,   0,   16,   18,   0,   0,   0,   0]]
    // CHECK:      {{\[}}[8,   4,   0,   0,   0,   0,   0,   5],
    // CHECK-NEXT: [4,   0,   0,   0,   0,   0,   3,   6],
    // CHECK-NEXT: [0,   2,   8,   0,   10,   0,   0,   13],
    // CHECK-NEXT: [14,   0,   16,   18,   0,   0,   0,   0]]
    // CHECK:      {{\[}}[12,   0,   0,   0,   0,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   12],
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   0]]
    //
    call @dump(%sm1) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump(%sm2) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump(%0) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump(%1) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump(%2) : (tensor<?x?xf64, #DCSR>) -> ()
    call @dump(%3) : (tensor<?x?xf64, #DCSR>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sm1 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %sm1_dup : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %sm2 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<?x?xf64, #DCSR>
    return
  }
}
