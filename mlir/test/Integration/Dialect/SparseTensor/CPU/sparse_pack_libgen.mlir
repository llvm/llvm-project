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
// Do the same run, but now with VLA vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=true vl=4
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

// TODO: This is considered to be a short-living tests and should be merged with sparse_pack.mlir
// after sparse_tensor.unpack is supported on libgen path.

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#SortedCOOI32 = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
  posWidth = 32,
  crdWidth = 32
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

// TODO: "compressed_hi" is not supported by libgen path.
// #BCOO = #sparse_tensor.encoding<{
//   map = (d0, d1, d2) -> (d0 : dense, d1 : compressed(nonunique, high), d2 : singleton)
//}>

module {
  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f64
    %i0 = arith.constant 0 : i32
    //
    // Initialize a 3-dim dense tensor.
    //
    %data = arith.constant dense<
       [  1.0,  2.0,  3.0]
    > : tensor<3xf64>

    %pos = arith.constant dense<
       [0, 3]
    > : tensor<2xindex>

    %index = arith.constant dense<
       [[  1,  2],
        [  5,  6],
        [  7,  8]]
    > : tensor<3x2xindex>

    %pos32 = arith.constant dense<
       [0, 3]
    > : tensor<2xi32>

    %index32 = arith.constant dense<
       [[  1,  2],
        [  5,  6],
        [  7,  8]]
    > : tensor<3x2xi32>

    %s4 = sparse_tensor.pack %data, %pos, %index : tensor<3xf64>, tensor<2xindex>, tensor<3x2xindex>
                                          to tensor<10x10xf64, #SortedCOO>
    %s5= sparse_tensor.pack %data, %pos32, %index32 : tensor<3xf64>, tensor<2xi32>, tensor<3x2xi32>
                                           to tensor<10x10xf64, #SortedCOOI32>

    %csr_data = arith.constant dense<
       [  1.0,  2.0,  3.0,  4.0]
    > : tensor<4xf64>

    %csr_pos32 = arith.constant dense<
       [0, 1, 3]
    > : tensor<3xi32>

    %csr_index32 = arith.constant dense<
       [1, 0, 1]
    > : tensor<3xi32>
    %csr= sparse_tensor.pack %csr_data, %csr_pos32, %csr_index32 : tensor<4xf64>, tensor<3xi32>, tensor<3xi32>
                                           to tensor<2x2xf64, #CSR>

    // CHECK:1
    // CHECK-NEXT:2
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:5
    // CHECK-NEXT:6
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:7
    // CHECK-NEXT:8
    // CHECK-NEXT:3
    sparse_tensor.foreach in %s4 : tensor<10x10xf64, #SortedCOO> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }

    // CHECK-NEXT:1
    // CHECK-NEXT:2
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:5
    // CHECK-NEXT:6
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:7
    // CHECK-NEXT:8
    // CHECK-NEXT:3
    sparse_tensor.foreach in %s5 : tensor<10x10xf64, #SortedCOOI32> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }

    // CHECK-NEXT:0
    // CHECK-NEXT:1
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:1
    // CHECK-NEXT:0
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:1
    // CHECK-NEXT:1
    // CHECK-NEXT:3
    sparse_tensor.foreach in %csr : tensor<2x2xf64, #CSR> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }


    bufferization.dealloc_tensor %s4  : tensor<10x10xf64, #SortedCOO>
    bufferization.dealloc_tensor %s5  : tensor<10x10xf64, #SortedCOOI32>
    bufferization.dealloc_tensor %csr  : tensor<2x2xf64, #CSR>

    return
  }
}
