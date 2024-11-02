//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
// do not use these LIT config files. Hence why this is kept inline.
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
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s

#CCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed),
  posWidth = 64,
  crdWidth = 32
}>

#BatchedCSR = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed),
  posWidth = 64,
  crdWidth = 32
}>

#CSRDense = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : dense),
  posWidth = 64,
  crdWidth = 32
}>

//
// Test assembly operation with CCC, batched-CSR and CSR-dense.
//
module {
  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32

    //
    // Setup CCC.
    //

    %data0 = arith.constant dense<
       [ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0 ]> : tensor<8xf32>
    %pos00 = arith.constant dense<
       [ 0, 3  ]> : tensor<2xi64>
    %crd00 = arith.constant dense<
       [ 0, 2, 3 ]> : tensor<3xi32>
    %pos01 = arith.constant dense<
       [ 0, 2, 4, 5  ]> : tensor<4xi64>
    %crd01 = arith.constant dense<
       [ 0, 1, 1, 2, 1 ]> : tensor<5xi32>
    %pos02 = arith.constant dense<
       [ 0, 2, 4, 5, 7, 8  ]> : tensor<6xi64>
    %crd02 = arith.constant dense<
       [ 0, 1, 0, 1, 0, 0, 1, 0  ]> : tensor<8xi32>

    %s0 = sparse_tensor.assemble (%pos00, %crd00, %pos01, %crd01, %pos02, %crd02), %data0 :
       (tensor<2xi64>, tensor<3xi32>,
        tensor<4xi64>, tensor<5xi32>,
        tensor<6xi64>, tensor<8xi32>), tensor<8xf32> to tensor<4x3x2xf32, #CCC>

    //
    // Setup BatchedCSR.
    //

    %data1 = arith.constant dense<
       [ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 ]> : tensor<16xf32>
    %pos1 = arith.constant dense<
       [ 0, 2, 3, 4, 6, 6, 7, 9, 11, 13, 14, 15, 16 ]> : tensor<13xi64>
    %crd1 = arith.constant dense<
       [ 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]> : tensor<16xi32>

    %s1 = sparse_tensor.assemble (%pos1, %crd1), %data1 : (tensor<13xi64>, tensor<16xi32>), tensor<16xf32> to tensor<4x3x2xf32, #BatchedCSR>

    //
    // Setup CSRDense.
    //

    %data2 = arith.constant dense<
      [ 1.0,  2.0,  0.0,  3.0,  4.0,  0.0, 5.0, 6.0,  0.0, 7.0,  8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 0.0, 15.0, 0.0, 16.0 ]> : tensor<22xf32>
    %pos2 = arith.constant dense<
      [ 0, 3, 5, 8, 11 ]> : tensor<5xi64>
    %crd2 = arith.constant dense<
      [ 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2 ]> : tensor<11xi32>

    %s2 = sparse_tensor.assemble (%pos2, %crd2), %data2  : (tensor<5xi64>, tensor<11xi32>), tensor<22xf32> to tensor<4x3x2xf32, #CSRDense>

    //
    // Verify.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 4, 3, 2 )
    // CHECK-NEXT: lvl = ( 4, 3, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 5
    // CHECK-NEXT: crd[1] : ( 0, 1, 1, 2, 1
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 5, 7, 8
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 0, 1, 0
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 16
    // CHECK-NEXT: dim = ( 4, 3, 2 )
    // CHECK-NEXT: lvl = ( 4, 3, 2 )
    // CHECK-NEXT: pos[2] : ( 0, 2, 3, 4, 6, 6, 7, 9, 11, 13, 14, 15, 16
    // CHECK-NEXT: crd[2] : ( 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 22
    // CHECK-NEXT: dim = ( 4, 3, 2 )
    // CHECK-NEXT: lvl = ( 4, 3, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 8, 11
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: values : ( 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 15, 0, 16
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %s0 : tensor<4x3x2xf32, #CCC>
    sparse_tensor.print %s1 : tensor<4x3x2xf32, #BatchedCSR>
    sparse_tensor.print %s2 : tensor<4x3x2xf32, #CSRDense>

    // FIXME: doing this explicitly crashes runtime
    // bufferization.dealloc_tensor %s0 : tensor<4x3x2xf32, #CCC>
    // bufferization.dealloc_tensor %s1 : tensor<4x3x2xf32, #BatchedCSR>
    // bufferization.dealloc_tensor %s2 : tensor<4x3x2xf32, #CSRDense>
    return
  }
}
