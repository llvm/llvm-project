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

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // a
    affine_map<(i) -> (i)>  // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = OP a(i)"
}

module {
  func.func @sparse_absf(%arg0: tensor<?xf64, #SparseVector>)
                             -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arg0, %c0 : tensor<?xf64, #SparseVector>
    %xin = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op
      ins(%arg0: tensor<?xf64, #SparseVector>)
      outs(%xin: tensor<?xf64, #SparseVector>) {
      ^bb0(%a: f64, %x: f64) :
        %result = math.absf %a : f64
        linalg.yield %result : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @sparse_absi(%arg0: tensor<?xi32, #SparseVector>)
                             -> tensor<?xi32, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arg0, %c0 : tensor<?xi32, #SparseVector>
    %xin = tensor.empty(%d) : tensor<?xi32, #SparseVector>
    %0 = linalg.generic #trait_op
      ins(%arg0: tensor<?xi32, #SparseVector>)
      outs(%xin: tensor<?xi32, #SparseVector>) {
      ^bb0(%a: i32, %x: i32) :
        %result = math.absi %a : i32
        linalg.yield %result : i32
    } -> tensor<?xi32, #SparseVector>
    return %0 : tensor<?xi32, #SparseVector>
  }

  // Driver method to call and verify sign kernel.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %df = arith.constant 99.99 : f64
    %di = arith.constant 9999 : i32

    %pnan = arith.constant 0x7FF0000001000000 : f64
    %nnan = arith.constant 0xFFF0000001000000 : f64
    %pinf = arith.constant 0x7FF0000000000000 : f64
    %ninf = arith.constant 0xFFF0000000000000 : f64

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [5], [11], [13], [17], [18], [20], [21], [28], [29], [31] ],
         [ -1.5, 1.5, -10.2, 11.3, 1.0, -1.0,
           0x7FF0000001000000, // +NaN
           0xFFF0000001000000, // -NaN
           0x7FF0000000000000, // +Inf
           0xFFF0000000000000, // -Inf
           -0.0,               // -Zero
           0.0                 // +Zero
        ]
    > : tensor<32xf64>
    %v2 = arith.constant sparse<
       [ [0], [3], [5], [11], [13], [17], [18], [21], [31] ],
         [ -2147483648, -2147483647, -1000, -1, 0,
           1, 1000, 2147483646, 2147483647
         ]
    > : tensor<32xi32>
    %sv1 = sparse_tensor.convert %v1
         : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %sv2 = sparse_tensor.convert %v2
         : tensor<32xi32> to tensor<?xi32, #SparseVector>

    // Call abs kernels.
    %0 = call @sparse_absf(%sv1) : (tensor<?xf64, #SparseVector>)
                                 -> tensor<?xf64, #SparseVector>

    %1 = call @sparse_absi(%sv2) : (tensor<?xi32, #SparseVector>)
                                 -> tensor<?xi32, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 12 )
    // CHECK-NEXT: crd[0] : ( 0, 3, 5, 11, 13, 17, 18, 20, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 1.5, 1.5, 10.2, 11.3, 1, 1, nan, nan, inf, inf, 0, 0 )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 9 )
    // CHECK-NEXT: crd[0] : ( 0, 3, 5, 11, 13, 17, 18, 21, 31 )
    // CHECK-NEXT: values : ( -2147483648, 2147483647, 1000, 1, 0, 1, 1000, 2147483646, 2147483647 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %1 : tensor<?xi32, #SparseVector>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xi32, #SparseVector>
    return
  }
}
