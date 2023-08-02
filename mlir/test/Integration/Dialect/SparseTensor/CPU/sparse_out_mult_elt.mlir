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

#DCSR = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ]
}>

#trait_mult_elt = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * B(i,j)"
}

module {
  // Sparse kernel.
  func.func @sparse_mult_elt(
      %arga: tensor<32x16xf32, #DCSR>, %argb: tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR> {
    %argx = bufferization.alloc_tensor() : tensor<32x16xf32, #DCSR>
    %0 = linalg.generic #trait_mult_elt
      ins(%arga, %argb: tensor<32x16xf32, #DCSR>, tensor<32x16xf32, #DCSR>)
      outs(%argx: tensor<32x16xf32, #DCSR>) {
        ^bb(%a: f32, %b: f32, %x: f32):
          %1 = arith.mulf %a, %b : f32
          linalg.yield %1 : f32
    } -> tensor<32x16xf32, #DCSR>
    return %0 : tensor<32x16xf32, #DCSR>
  }

  // Driver method to call and verify kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32

    // Setup very sparse matrices.
    %ta = arith.constant sparse<
       [ [2,2], [15,15], [31,0], [31,14] ], [ 2.0, 3.0, -2.0, 4.0 ]
    > : tensor<32x16xf32>
    %tb = arith.constant sparse<
       [ [1,1], [2,0], [2,2], [2,15], [31,0], [31,15] ], [ 5.0, 6.0, 7.0, 8.0, -10.0, 9.0 ]
    > : tensor<32x16xf32>
    %sta = sparse_tensor.convert %ta
      : tensor<32x16xf32> to tensor<32x16xf32, #DCSR>
    %stb = sparse_tensor.convert %tb
      : tensor<32x16xf32> to tensor<32x16xf32, #DCSR>

    // Call kernel.
    %0 = call @sparse_mult_elt(%sta, %stb)
      : (tensor<32x16xf32, #DCSR>,
         tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR>

    //
    // Verify results. Only two entries stored in result!
    //
    // CHECK: ( 14, 20, 0, 0 )
    //
    %val = sparse_tensor.values %0 : tensor<32x16xf32, #DCSR> to memref<?xf32>
    %vv = vector.transfer_read %val[%c0], %f0: memref<?xf32>, vector<4xf32>
    vector.print %vv : vector<4xf32>

    // Release the resources.
    bufferization.dealloc_tensor %sta : tensor<32x16xf32, #DCSR>
    bufferization.dealloc_tensor %stb : tensor<32x16xf32, #DCSR>
    bufferization.dealloc_tensor %0   : tensor<32x16xf32, #DCSR>
    return
  }
}
