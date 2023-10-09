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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
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
