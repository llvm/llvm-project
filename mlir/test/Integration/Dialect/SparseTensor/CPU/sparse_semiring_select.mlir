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

#DCSR = #sparse_tensor.encoding<{
  lvlTypes = ["compressed", "compressed"]
}>

#sel_trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // C (in)
    affine_map<(i,j) -> (i,j)>,  // L (in)
    affine_map<(i,j) -> (i,j)>,  // R (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

module {
  func.func @sparse_select(%cond: tensor<5x5xi1>,
                           %arga: tensor<5x5xf64, #DCSR>,
                           %argb: tensor<5x5xf64, #DCSR>) -> tensor<5x5xf64, #DCSR> {
    %xv = bufferization.alloc_tensor() : tensor<5x5xf64, #DCSR>
    %0 = linalg.generic #sel_trait
       ins(%cond, %arga, %argb: tensor<5x5xi1>, tensor<5x5xf64, #DCSR>, tensor<5x5xf64, #DCSR>)
        outs(%xv: tensor<5x5xf64, #DCSR>) {
        ^bb(%c: i1, %a: f64, %b: f64, %x: f64):
          %1 = arith.select %c, %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<5x5xf64, #DCSR>
    return %0 : tensor<5x5xf64, #DCSR>
  }

  // Driver method to call and verify vector kernels.
  func.func @entry() {
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64

    %cond = arith.constant sparse<
        [ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4] ],
        [     1,      1,      1,      1,      1  ]
    > : tensor<5x5xi1>
    %lhs = arith.constant sparse<
        [ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4] ],
        [   0.1,    1.1,    2.1,    3.1,    4.1  ]
    > : tensor<5x5xf64>
    %rhs = arith.constant sparse<
        [ [0, 1], [1, 2], [2, 3], [3, 4], [4, 4]],
        [   1.1,    2.2,    3.3,    4.4 ,   5.5 ]
    > : tensor<5x5xf64>

    %sl = sparse_tensor.convert %lhs : tensor<5x5xf64> to tensor<5x5xf64, #DCSR>
    %sr = sparse_tensor.convert %rhs : tensor<5x5xf64> to tensor<5x5xf64, #DCSR>

    // Call sparse matrix kernels.
    %1 = call @sparse_select(%cond, %sl, %sr) : (tensor<5x5xi1>,
                                                 tensor<5x5xf64, #DCSR>,
                                                 tensor<5x5xf64, #DCSR>) -> tensor<5x5xf64, #DCSR>


    // CHECK:     ( ( 0.1, 1.1, 0, 0, 0 ),
    // CHECK-SAME:  ( 0, 1.1, 2.2, 0, 0 ),
    // CHECK-SAME:  ( 0, 0, 2.1, 3.3, 0 ),
    // CHECK-SAME:  ( 0, 0, 0, 3.1, 4.4 ),
    // CHECK-SAME:  ( 0, 0, 0, 0, 4.1 ) )
    %r = sparse_tensor.convert %1 : tensor<5x5xf64, #DCSR> to tensor<5x5xf64>
    %v2 = vector.transfer_read %r[%c0, %c0], %f0 : tensor<5x5xf64>, vector<5x5xf64>
    vector.print %v2 : vector<5x5xf64>

    // Release the resources.
    bufferization.dealloc_tensor %sl: tensor<5x5xf64, #DCSR>
    bufferization.dealloc_tensor %sr: tensor<5x5xf64, #DCSR>
    bufferization.dealloc_tensor %1:  tensor<5x5xf64, #DCSR>

    return
  }
}
