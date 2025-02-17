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
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait_reduction = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> ()>    // x (scalar out)
  ],
  iterator_types = ["reduction"],
  doc = "x += SUM_CUSTOM_i UNARY a(i)"
}

// Test of unary op feeding into custom sum reduction.
module {

  // Contrived example for stress testing, where neither branch feeds
  // a value into a subsequent custom sum reduction. The code should
  // be folded into the initial value 1.
  func.func @red0(%arga: tensor<8xi32, #SV>, %argx: tensor<i32>) -> tensor<i32> {
    %c1 = arith.constant 1 : i32
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<8xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
           %u = sparse_tensor.unary %a : i32 to i32
             present={ }
             absent={ }
           %r = sparse_tensor.reduce %u, %b, %c1 : i32 {
            ^bb0(%x: i32, %y: i32):
              %sum = arith.addi %x, %y : i32
              sparse_tensor.yield %sum : i32
           }
        linalg.yield %r : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  // Typical example where present branch contributes a value
  // into a subsequent custom sum reduction.
  func.func @red1(%arga: tensor<8xi32, #SV>, %argx: tensor<i32>) -> tensor<i32> {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<8xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
           %u = sparse_tensor.unary %a : i32 to i32
             present={
               ^bb(%p: i32):
                sparse_tensor.yield %c2 : i32
             }
             absent={ }
           %r = sparse_tensor.reduce %u, %b, %c1 : i32 {
            ^bb0(%x: i32, %y: i32):
              %sum = arith.addi %x, %y : i32
              sparse_tensor.yield %sum : i32
           }
        linalg.yield %r : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  // A complementing example where absent branch contributes a value
  // into a subsequent custom sum reduction.
  func.func @red2(%arga: tensor<8xi32, #SV>, %argx: tensor<i32>) -> tensor<i32> {
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<8xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
           %u = sparse_tensor.unary %a : i32 to i32
           present={ }
           absent={
               sparse_tensor.yield %c3 : i32
          }
          %r = sparse_tensor.reduce %u, %b, %c1 : i32 {
            ^bb0(%x: i32, %y: i32):
              %sum = arith.addi %x, %y : i32
              sparse_tensor.yield %sum : i32
          }
        linalg.yield %r : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  // An example where both present and absent branch contribute values
  // into a subsequent custom sum reduction.
  func.func @red3(%arga: tensor<8xi32, #SV>, %argx: tensor<i32>) -> tensor<i32> {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<8xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
           %u = sparse_tensor.unary %a : i32 to i32
           present={
             ^bb(%p: i32):
               sparse_tensor.yield %c2 : i32
           } absent={
               sparse_tensor.yield %c3 : i32
          }
          %r = sparse_tensor.reduce %u, %b, %c1 : i32 {
            ^bb0(%x: i32, %y: i32):
              %sum = arith.addi %x, %y : i32
              sparse_tensor.yield %sum : i32
          }
          linalg.yield %r : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @dump_i32(%arg0 : tensor<i32>) {
    %v = tensor.extract %arg0[] : tensor<i32>
    vector.print %v : i32
    return
  }

  func.func @main() {
    %ri = arith.constant dense<0> : tensor<i32>

    //  Sparse vector of length 8 with 2 stored elements (and thus 6 implicit zeros).
    %v0 = arith.constant sparse< [ [4], [6] ], [ 99, 999 ] > : tensor<8xi32>
    %s0 = sparse_tensor.convert %v0: tensor<8xi32> to tensor<8xi32, #SV>

    // Call the kernels.
    %0 = call @red0(%s0, %ri) : (tensor<8xi32, #SV>, tensor<i32>) -> tensor<i32>
    %1 = call @red1(%s0, %ri) : (tensor<8xi32, #SV>, tensor<i32>) -> tensor<i32>
    %2 = call @red2(%s0, %ri) : (tensor<8xi32, #SV>, tensor<i32>) -> tensor<i32>
    %3 = call @red3(%s0, %ri) : (tensor<8xi32, #SV>, tensor<i32>) -> tensor<i32>

    // Verify results.
    //   1 + nothing
    //   1 + 2 x present
    //   1 + 3 x absent
    //   1 + 2 x present + 3 x absent
    //
    // CHECK: 1
    // CHECK: 5
    // CHECK: 19
    // CHECK: 23
    //
    call @dump_i32(%0) : (tensor<i32>) -> ()
    call @dump_i32(%1) : (tensor<i32>) -> ()
    call @dump_i32(%2) : (tensor<i32>) -> ()
    call @dump_i32(%3) : (tensor<i32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %s0 : tensor<8xi32, #SV>
    bufferization.dealloc_tensor %0 : tensor<i32>
    bufferization.dealloc_tensor %1 : tensor<i32>
    bufferization.dealloc_tensor %2 : tensor<i32>
    bufferization.dealloc_tensor %3 : tensor<i32>

    return
  }
}
