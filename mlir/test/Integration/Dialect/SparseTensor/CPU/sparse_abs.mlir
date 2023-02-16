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

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

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
    %xin = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
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
    %xin = bufferization.alloc_tensor(%d) : tensor<?xi32, #SparseVector>
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
  func.func @entry() {
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
    // CHECK:       12
    // CHECK-NEXT: ( 1.5, 1.5, 10.2, 11.3, 1, 1, nan, nan, inf, inf, 0, 0 )
    // CHECK-NEXT:  9
    // CHECK-NEXT: ( -2147483648, 2147483647, 1000, 1, 0, 1, 1000, 2147483646, 2147483647 )
    //
    %x = sparse_tensor.values %0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %y = sparse_tensor.values %1 : tensor<?xi32, #SparseVector> to memref<?xi32>
    %a = vector.transfer_read %x[%c0], %df: memref<?xf64>, vector<12xf64>
    %b = vector.transfer_read %y[%c0], %di: memref<?xi32>, vector<9xi32>
    %na = sparse_tensor.number_of_entries %0 : tensor<?xf64, #SparseVector>
    %nb = sparse_tensor.number_of_entries %1 : tensor<?xi32, #SparseVector>
    vector.print %na : index
    vector.print %a : vector<12xf64>
    vector.print %nb : index
    vector.print %b : vector<9xi32>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xi32, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xi32, #SparseVector>
    return
  }
}
