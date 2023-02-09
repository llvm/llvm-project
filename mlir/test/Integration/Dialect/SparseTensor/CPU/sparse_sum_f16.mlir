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

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#trait_sum_reduce = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> ()>     // x (out)
  ],
  iterator_types = ["reduction", "reduction"],
  doc = "x += A(i,j)"
}

module {
  //
  // A kernel that sum-reduces a matrix to a single scalar.
  //
  func.func @kernel_sum_reduce(%arga: tensor<?x?xf16, #SparseMatrix>,
                               %argx: tensor<f16>) -> tensor<f16> {
    %0 = linalg.generic #trait_sum_reduce
      ins(%arga: tensor<?x?xf16, #SparseMatrix>)
      outs(%argx: tensor<f16>) {
      ^bb(%a: f16, %x: f16):
        %0 = arith.addf %x, %a : f16
        linalg.yield %0 : f16
    } -> tensor<f16>
    return %0 : tensor<f16>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    // Setup input sparse matrix from compressed constant.
    %d = arith.constant dense <[
       [ 1.1,  1.2,  0.0,  1.4 ],
       [ 0.0,  0.0,  0.0,  0.0 ],
       [ 3.1,  0.0,  3.3,  3.4 ]
    ]> : tensor<3x4xf16>
    %a = sparse_tensor.convert %d : tensor<3x4xf16> to tensor<?x?xf16, #SparseMatrix>

    %d0 = arith.constant 0.0 : f16
    // Setup memory for a single reduction scalar,
    // initialized to zero.
    %x = tensor.from_elements %d0 : tensor<f16>

    // Call the kernel.
    %0 = call @kernel_sum_reduce(%a, %x)
      : (tensor<?x?xf16, #SparseMatrix>, tensor<f16>) -> tensor<f16>

    // Print the result for verification.
    //
    // CHECK: 13.5
    //
    %v = tensor.extract %0[] : tensor<f16>
    %vf = arith.extf %v: f16 to f32
    vector.print %vf : f32

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xf16, #SparseMatrix>

    return
  }
}
