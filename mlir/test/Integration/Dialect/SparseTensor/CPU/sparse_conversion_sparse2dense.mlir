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
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#Tensor1  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#Tensor2  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d1 : compressed, d2 : compressed, d0 : compressed),
}>

#Tensor3  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed, d0 : compressed, d1 : compressed),
}>

#Tensor4  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

#Tensor5  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d1 : dense, d2 : compressed, d0 : compressed)
}>

#Tensor6  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

//
// Integration test that tests conversions from sparse to dense tensors.
//
module {
  //
  // Utilities for output.
  //
  func.func @dump_234(%arg0: tensor<2x3x4xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %d0: tensor<2x3x4xf64>, vector<2x3x4xf64>
    vector.print %0 : vector<2x3x4xf64>
    return
  }
  func.func @dump_p34(%arg0: tensor<?x3x4xf64>) {
    %0 = tensor.cast %arg0 : tensor<?x3x4xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }
  func.func @dump_2p4(%arg0: tensor<2x?x4xf64>) {
    %0 = tensor.cast %arg0 : tensor<2x?x4xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }
  func.func @dump_23p(%arg0: tensor<2x3x?xf64>) {
    %0 = tensor.cast %arg0 : tensor<2x3x?xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }
  func.func @dump_2pp(%arg0: tensor<2x?x?xf64>) {
    %0 = tensor.cast %arg0 : tensor<2x?x?xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }
  func.func @dump_p3p(%arg0: tensor<?x3x?xf64>) {
    %0 = tensor.cast %arg0 : tensor<?x3x?xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }
  func.func @dump_pp4(%arg0: tensor<?x?x4xf64>) {
    %0 = tensor.cast %arg0 : tensor<?x?x4xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }
  func.func @dump_ppp(%arg0: tensor<?x?x?xf64>) {
    %0 = tensor.cast %arg0 : tensor<?x?x?xf64> to tensor<2x3x4xf64>
    call @dump_234(%0) : (tensor<2x3x4xf64>) -> ()
    return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    //
    // Initialize a 3-dim dense tensor.
    //
    %src = arith.constant dense<[
       [  [  1.0,  2.0,  3.0,  4.0 ],
          [  5.0,  6.0,  7.0,  8.0 ],
          [  9.0, 10.0, 11.0, 12.0 ] ],
       [  [ 13.0, 14.0, 15.0, 16.0 ],
          [ 17.0, 18.0, 19.0, 20.0 ],
          [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //
    %s2341 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor1>
    %s2342 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor2>
    %s2343 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor3>
    %s2344 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor4>
    %s2345 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor5>
    %s2346 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor6>

    // NOTE: We commented out most cases with the dynamic-sized output tensor because finishing
    // all of them are currently taking too long, and they are not covering too many new code
    // paths.
    %sp344 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<?x3x4xf64, #Tensor4>
    // %sp345 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<?x3x4xf64, #Tensor5>
    // %sp346 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<?x3x4xf64, #Tensor6>
    // %s2p44 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x?x4xf64, #Tensor4>
    %s2p45 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x?x4xf64, #Tensor5>
    // %s2p46 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x?x4xf64, #Tensor6>
    // %s23p4 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x?xf64, #Tensor4>
    // %s23p5 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x?xf64, #Tensor5>
    %s23p6 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x3x?xf64, #Tensor6>
    // %s2pp4 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x?x?xf64, #Tensor4>
    // %s2pp5 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x?x?xf64, #Tensor5>
    // %s2pp6 = sparse_tensor.convert %src : tensor<2x3x4xf64> to tensor<2x?x?xf64, #Tensor6>

    //
    // Convert sparse tensor back to dense.
    //
    %d2341 = sparse_tensor.convert %s2341 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64>
    %d2342 = sparse_tensor.convert %s2342 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64>
    %d2343 = sparse_tensor.convert %s2343 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64>
    %d2344 = sparse_tensor.convert %s2344 : tensor<2x3x4xf64, #Tensor4> to tensor<2x3x4xf64>
    %d2345 = sparse_tensor.convert %s2345 : tensor<2x3x4xf64, #Tensor5> to tensor<2x3x4xf64>
    %d2346 = sparse_tensor.convert %s2346 : tensor<2x3x4xf64, #Tensor6> to tensor<2x3x4xf64>

    %dp344 = sparse_tensor.convert %sp344 : tensor<?x3x4xf64, #Tensor4> to tensor<?x3x4xf64>
    // %dp345 = sparse_tensor.convert %sp345 : tensor<?x3x4xf64, #Tensor5> to tensor<?x3x4xf64>
    // %dp346 = sparse_tensor.convert %sp346 : tensor<?x3x4xf64, #Tensor6> to tensor<?x3x4xf64>
    // %d2p44 = sparse_tensor.convert %s2p44 : tensor<2x?x4xf64, #Tensor4> to tensor<2x?x4xf64>
    %d2p45 = sparse_tensor.convert %s2p45 : tensor<2x?x4xf64, #Tensor5> to tensor<2x?x4xf64>
    // %d2p46 = sparse_tensor.convert %s2p46 : tensor<2x?x4xf64, #Tensor6> to tensor<2x?x4xf64>
    // %d23p4 = sparse_tensor.convert %s23p4 : tensor<2x3x?xf64, #Tensor4> to tensor<2x3x?xf64>
    // %d23p5 = sparse_tensor.convert %s23p5 : tensor<2x3x?xf64, #Tensor5> to tensor<2x3x?xf64>
    %d23p6 = sparse_tensor.convert %s23p6 : tensor<2x3x?xf64, #Tensor6> to tensor<2x3x?xf64>
    // %d2pp4 = sparse_tensor.convert %s2pp4 : tensor<2x?x?xf64, #Tensor4> to tensor<2x?x?xf64>
    // %d2pp5 = sparse_tensor.convert %s2pp5 : tensor<2x?x?xf64, #Tensor5> to tensor<2x?x?xf64>
    // %d2pp6 = sparse_tensor.convert %s2pp6 : tensor<2x?x?xf64, #Tensor6> to tensor<2x?x?xf64>

    %dp3p4 = sparse_tensor.convert %sp344 : tensor<?x3x4xf64, #Tensor4> to tensor<?x3x?xf64>
    // %dp3p5 = sparse_tensor.convert %sp345 : tensor<?x3x4xf64, #Tensor5> to tensor<?x3x?xf64>
    // %dp3p6 = sparse_tensor.convert %sp346 : tensor<?x3x4xf64, #Tensor6> to tensor<?x3x?xf64>
    // %dpp44 = sparse_tensor.convert %s2p44 : tensor<2x?x4xf64, #Tensor4> to tensor<?x?x4xf64>
    %dpp45 = sparse_tensor.convert %s2p45 : tensor<2x?x4xf64, #Tensor5> to tensor<?x?x4xf64>
    // %dpp46 = sparse_tensor.convert %s2p46 : tensor<2x?x4xf64, #Tensor6> to tensor<?x?x4xf64>
    // %dppp4 = sparse_tensor.convert %s2pp4 : tensor<2x?x?xf64, #Tensor4> to tensor<?x?x?xf64>
    // %dppp5 = sparse_tensor.convert %s2pp5 : tensor<2x?x?xf64, #Tensor5> to tensor<?x?x?xf64>
    // %dppp6 = sparse_tensor.convert %s2pp6 : tensor<2x?x?xf64, #Tensor6> to tensor<?x?x?xf64>

    //
    // Check round-trip equality.  And release dense tensors.
    // CHECK-COUNT-12: ( ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ), ( 9, 10, 11, 12 ) ), ( ( 13, 14, 15, 16 ), ( 17, 18, 19, 20 ), ( 21, 22, 23, 24 ) ) )
    //   was COUNT-28 before.
    call @dump_234(%src) : (tensor<2x3x4xf64>) -> ()
    call @dump_234(%d2341) : (tensor<2x3x4xf64>) -> ()
    call @dump_234(%d2342) : (tensor<2x3x4xf64>) -> ()
    call @dump_234(%d2343) : (tensor<2x3x4xf64>) -> ()
    call @dump_234(%d2344) : (tensor<2x3x4xf64>) -> ()
    call @dump_234(%d2345) : (tensor<2x3x4xf64>) -> ()
    call @dump_234(%d2346) : (tensor<2x3x4xf64>) -> ()
    call @dump_p34(%dp344) : (tensor<?x3x4xf64>) -> ()
    // call @dump_p34(%dp345) : (tensor<?x3x4xf64>) -> ()
    // call @dump_p34(%dp346) : (tensor<?x3x4xf64>) -> ()
    // call @dump_2p4(%d2p44) : (tensor<2x?x4xf64>) -> ()
    call @dump_2p4(%d2p45) : (tensor<2x?x4xf64>) -> ()
    // call @dump_2p4(%d2p46) : (tensor<2x?x4xf64>) -> ()
    // call @dump_23p(%d23p4) : (tensor<2x3x?xf64>) -> ()
    // call @dump_23p(%d23p5) : (tensor<2x3x?xf64>) -> ()
    call @dump_23p(%d23p6) : (tensor<2x3x?xf64>) -> ()
    // call @dump_2pp(%d2pp4) : (tensor<2x?x?xf64>) -> ()
    // call @dump_2pp(%d2pp5) : (tensor<2x?x?xf64>) -> ()
    // call @dump_2pp(%d2pp6) : (tensor<2x?x?xf64>) -> ()
    call @dump_p3p(%dp3p4) : (tensor<?x3x?xf64>) -> ()
    // call @dump_p3p(%dp3p5) : (tensor<?x3x?xf64>) -> ()
    // call @dump_p3p(%dp3p6) : (tensor<?x3x?xf64>) -> ()
    // call @dump_pp4(%dpp44) : (tensor<?x?x4xf64>) -> ()
    call @dump_pp4(%dpp45) : (tensor<?x?x4xf64>) -> ()
    // call @dump_pp4(%dpp46) : (tensor<?x?x4xf64>) -> ()
    // call @dump_ppp(%dppp4) : (tensor<?x?x?xf64>) -> ()
    // call @dump_ppp(%dppp5) : (tensor<?x?x?xf64>) -> ()
    // call @dump_ppp(%dppp6) : (tensor<?x?x?xf64>) -> ()

    //
    // Release sparse tensors.
    //
    bufferization.dealloc_tensor %s2341 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %s2342 : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %s2343 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %s2344 : tensor<2x3x4xf64, #Tensor4>
    bufferization.dealloc_tensor %s2345 : tensor<2x3x4xf64, #Tensor5>
    bufferization.dealloc_tensor %s2346 : tensor<2x3x4xf64, #Tensor6>
    bufferization.dealloc_tensor %sp344 : tensor<?x3x4xf64, #Tensor4>
    // bufferization.dealloc_tensor %sp345 : tensor<?x3x4xf64, #Tensor5>
    // bufferization.dealloc_tensor %sp346 : tensor<?x3x4xf64, #Tensor6>
    // bufferization.dealloc_tensor %s2p44 : tensor<2x?x4xf64, #Tensor4>
    bufferization.dealloc_tensor %s2p45 : tensor<2x?x4xf64, #Tensor5>
    // bufferization.dealloc_tensor %s2p46 : tensor<2x?x4xf64, #Tensor6>
    // bufferization.dealloc_tensor %s23p4 : tensor<2x3x?xf64, #Tensor4>
    // bufferization.dealloc_tensor %s23p5 : tensor<2x3x?xf64, #Tensor5>
    bufferization.dealloc_tensor %s23p6 : tensor<2x3x?xf64, #Tensor6>
    // bufferization.dealloc_tensor %s2pp4 : tensor<2x?x?xf64, #Tensor4>
    // bufferization.dealloc_tensor %s2pp5 : tensor<2x?x?xf64, #Tensor5>
    // bufferization.dealloc_tensor %s2pp6 : tensor<2x?x?xf64, #Tensor6>
    return
  }
}
