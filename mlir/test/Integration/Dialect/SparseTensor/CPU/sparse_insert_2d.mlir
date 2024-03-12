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

// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#Dense = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa))
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#Row = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : dense)
}>

module {
  //
  // Main driver. We test the contents of various sparse tensor
  // schemes when they are still empty and after a few insertions.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %f1 = arith.constant 1.0 : f64
    %f2 = arith.constant 2.0 : f64
    %f3 = arith.constant 3.0 : f64
    %f4 = arith.constant 4.0 : f64

    //
    // Dense case.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: values : ( 1, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 4,
    // CHECK-NEXT: ----
    //
    %densea = tensor.empty() : tensor<4x3xf64, #Dense>
    %dense1 = tensor.insert %f1 into %densea[%c0, %c0] : tensor<4x3xf64, #Dense>
    %dense2 = tensor.insert %f2 into %dense1[%c2, %c2] : tensor<4x3xf64, #Dense>
    %dense3 = tensor.insert %f3 into %dense2[%c3, %c0] : tensor<4x3xf64, #Dense>
    %dense4 = tensor.insert %f4 into %dense3[%c3, %c2] : tensor<4x3xf64, #Dense>
    %densem = sparse_tensor.load %dense4 hasInserts : tensor<4x3xf64, #Dense>
    sparse_tensor.print %densem : tensor<4x3xf64, #Dense>

    //
    // COO case.
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4,
    // CHECK-NEXT: crd[0] : ( 0, 2, 3, 3,
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2,
    // CHECK-NEXT: values : ( 1, 2, 3, 4,
    // CHECK-NEXT: ----
    //
    %cooa = tensor.empty() : tensor<4x3xf64, #SortedCOO>
    %coo1 = tensor.insert %f1 into %cooa[%c0, %c0] : tensor<4x3xf64, #SortedCOO>
    %coo2 = tensor.insert %f2 into %coo1[%c2, %c2] : tensor<4x3xf64, #SortedCOO>
    %coo3 = tensor.insert %f3 into %coo2[%c3, %c0] : tensor<4x3xf64, #SortedCOO>
    %coo4 = tensor.insert %f4 into %coo3[%c3, %c2] : tensor<4x3xf64, #SortedCOO>
    %coom = sparse_tensor.load %coo4 hasInserts : tensor<4x3xf64, #SortedCOO>
    sparse_tensor.print %coom : tensor<4x3xf64, #SortedCOO>

    //
    // CSR case.
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 1, 2, 4,
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2,
    // CHECK-NEXT: values : ( 1, 2, 3, 4,
    // CHECK-NEXT: ----
    //
    %csra = tensor.empty() : tensor<4x3xf64, #CSR>
    %csr1 = tensor.insert %f1 into %csra[%c0, %c0] : tensor<4x3xf64, #CSR>
    %csr2 = tensor.insert %f2 into %csr1[%c2, %c2] : tensor<4x3xf64, #CSR>
    %csr3 = tensor.insert %f3 into %csr2[%c3, %c0] : tensor<4x3xf64, #CSR>
    %csr4 = tensor.insert %f4 into %csr3[%c3, %c2] : tensor<4x3xf64, #CSR>
    %csrm = sparse_tensor.load %csr4 hasInserts : tensor<4x3xf64, #CSR>
    sparse_tensor.print %csrm : tensor<4x3xf64, #CSR>

    //
    // DCSR case.
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 3,
    // CHECK-NEXT: crd[0] : ( 0, 2, 3,
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 4,
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2,
    // CHECK-NEXT: values : ( 1, 2, 3, 4,
    // CHECK-NEXT: ----
    //
    %dcsra = tensor.empty() : tensor<4x3xf64, #DCSR>
    %dcsr1 = tensor.insert %f1 into %dcsra[%c0, %c0] : tensor<4x3xf64, #DCSR>
    %dcsr2 = tensor.insert %f2 into %dcsr1[%c2, %c2] : tensor<4x3xf64, #DCSR>
    %dcsr3 = tensor.insert %f3 into %dcsr2[%c3, %c0] : tensor<4x3xf64, #DCSR>
    %dcsr4 = tensor.insert %f4 into %dcsr3[%c3, %c2] : tensor<4x3xf64, #DCSR>
    %dcsrm = sparse_tensor.load %dcsr4 hasInserts : tensor<4x3xf64, #DCSR>
    sparse_tensor.print %dcsrm : tensor<4x3xf64, #DCSR>

    //
    // Row case.
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 3,
    // CHECK-NEXT: crd[0] : ( 0, 2, 3,
    // CHECK-NEXT: values : ( 1, 0, 0, 0, 0, 2, 3, 0, 4,
    // CHECK-NEXT: ----
    //
    %rowa = tensor.empty() : tensor<4x3xf64, #Row>
    %row1 = tensor.insert %f1 into %rowa[%c0, %c0] : tensor<4x3xf64, #Row>
    %row2 = tensor.insert %f2 into %row1[%c2, %c2] : tensor<4x3xf64, #Row>
    %row3 = tensor.insert %f3 into %row2[%c3, %c0] : tensor<4x3xf64, #Row>
    %row4 = tensor.insert %f4 into %row3[%c3, %c2] : tensor<4x3xf64, #Row>
    %rowm = sparse_tensor.load %row4 hasInserts : tensor<4x3xf64, #Row>
    sparse_tensor.print %rowm : tensor<4x3xf64, #Row>

    // Release resources.
    bufferization.dealloc_tensor %densem : tensor<4x3xf64, #Dense>
    bufferization.dealloc_tensor %coom : tensor<4x3xf64, #SortedCOO>
    bufferization.dealloc_tensor %csrm : tensor<4x3xf64, #CSR>
    bufferization.dealloc_tensor %dcsrm : tensor<4x3xf64, #DCSR>
    bufferization.dealloc_tensor %rowm : tensor<4x3xf64, #Row>

    return
  }
}
