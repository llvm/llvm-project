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
// Do the same run, but now do sparsification using sparse-iterator-based loops.
// REDEFINE: %{sparsifier_opts} = sparse-emit-strategy=sparse-iterator
// RUN: %{compile} | %{run} | FileCheck %s
//

#COO = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (
    d0 : compressed(nonunique),
    d1 : singleton(nonunique, soa),
    d2 : singleton(nonunique, soa),
    d3 : singleton(soa)
  ),
  explicitVal = 1 : i32
}>

// An example of vector reductions.
module {

  func.func @sqsum(%arg0: tensor<2x3x4x5xi32, #COO>) -> tensor<i32> {
    %cst = arith.constant dense<0> : tensor<i32>
    %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> ()>
      ],
      iterator_types = ["reduction", "reduction", "reduction", "reduction"]
    } ins(%arg0 : tensor<2x3x4x5xi32, #COO>) outs(%cst : tensor<i32>) {
    ^bb0(%in: i32, %out: i32):
      %1 = arith.muli %in, %in : i32
      %2 = arith.addi %out, %1 : i32
      linalg.yield %2 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @main() {
    %cst = arith.constant sparse<
     [
       [0, 1, 2, 3],
       [1, 1, 2, 3],
       [1, 2, 2, 3],
       [1, 2, 3, 4]
     ],
     [1, 1, 1, 1]
    > : tensor<2x3x4x5xi32>

    %input = sparse_tensor.convert %cst : tensor<2x3x4x5xi32> to tensor<2x3x4x5xi32, #COO>
    %0 = call @sqsum(%input) : (tensor<2x3x4x5xi32, #COO>) -> tensor<i32>
    %v = tensor.extract %0[] : tensor<i32>

    // CHECK: 4
    vector.print %v : i32

    bufferization.dealloc_tensor %input : tensor<2x3x4x5xi32, #COO>
    bufferization.dealloc_tensor %0 : tensor<i32>
    return
  }
}
