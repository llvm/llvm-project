// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// FIXME: lib path does not support all of COO yet
// R_U_N: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

module {

  // A linalg representation of some higher "transpose" op.
  func.func @transpose_coo(%arga: tensor<10x5xf32, #SortedCOO>)
                               -> tensor<5x10xf32, #SortedCOO> {
    %0 = bufferization.alloc_tensor() : tensor<5x10xf32, #SortedCOO>
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arga : tensor<10x5xf32, #SortedCOO>)
      outs(%0 : tensor<5x10xf32, #SortedCOO>) {
        ^bb0(%in: f32, %out: f32):
           linalg.yield %in : f32
      } -> tensor<5x10xf32, #SortedCOO>
    return %1 : tensor<5x10xf32, #SortedCOO>
  }

  func.func @entry() {
    %f0  = arith.constant 0.0 : f32
    %c0  = arith.constant 0   : index
    %c1  = arith.constant 1   : index

    %A = arith.constant dense<
        [ [ 10.0, 20.0, 30.0, 40.0, 50.0 ],
          [ 11.0, 21.0, 31.0, 41.0, 51.0 ],
          [ 12.0, 22.0, 32.0, 42.0, 52.0 ],
          [ 13.0, 23.0, 33.0, 43.0, 53.0 ],
          [ 14.0, 24.0, 34.0, 44.0, 54.0 ],
          [ 15.0, 25.0, 35.0, 45.0, 55.0 ],
          [ 16.0, 26.0, 36.0, 46.0, 56.0 ],
          [ 17.0, 27.0, 37.0, 47.0, 57.0 ],
          [ 18.0, 28.0, 38.0, 48.0, 58.0 ],
          [ 19.0, 29.0, 39.0, 49.0, 59.0 ] ]
    > : tensor<10x5xf32>

    // Stress test with a "sparse" version of A.
    %SA = sparse_tensor.convert %A
      : tensor<10x5xf32> to tensor<10x5xf32, #SortedCOO>
    %SAT = call @transpose_coo(%SA) : (tensor<10x5xf32, #SortedCOO>)
                                    -> tensor<5x10xf32, #SortedCOO>

    //
    // Verify original and transposed sorted COO.
    //
    // CHECK:      ( 10, 20, 30, 40, 50, 11, 21, 31, 41, 51, 12, 22, 32, 42, 52, 13, 23, 33, 43, 53, 14, 24, 34, 44, 54, 15, 25, 35, 45, 55, 16, 26, 36, 46, 56, 17, 27, 37, 47, 57, 18, 28, 38, 48, 58, 19, 29, 39, 49, 59 )
    // CHECK-NEXT: ( 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 )
    //
    %va  = sparse_tensor.values %SA
      : tensor<10x5xf32, #SortedCOO> to memref<?xf32>
    %vat = sparse_tensor.values %SAT
      : tensor<5x10xf32, #SortedCOO> to memref<?xf32>
    %v1 = vector.transfer_read %va[%c0],  %f0 : memref<?xf32>, vector<50xf32>
    %v2 = vector.transfer_read %vat[%c0], %f0 : memref<?xf32>, vector<50xf32>
    vector.print %v1 : vector<50xf32>
    vector.print %v2 : vector<50xf32>

    // Release resources.
    bufferization.dealloc_tensor %SA  : tensor<10x5xf32, #SortedCOO>
    bufferization.dealloc_tensor %SAT : tensor<5x10xf32, #SortedCOO>

    return
  }
}
