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
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
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
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#TensorCSR = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : dense, d2 : compressed)
}>

#TensorRow = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : dense)
}>

#CCoo = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed(nonunique), d2 : singleton)
}>

#DCoo = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed(nonunique), d2 : singleton)
}>


module {
  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %f1 = arith.constant 1.1 : f64
    %f2 = arith.constant 2.2 : f64
    %f3 = arith.constant 3.3 : f64
    %f4 = arith.constant 4.4 : f64
    %f5 = arith.constant 5.5 : f64

    // CHECK: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 5, 4, 3 )
    // CHECK-NEXT: lvl = ( 5, 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 3, 4 )
    // CHECK-NEXT: pos[2] : ( 0, 2, 2, 2, 3, 3, 3, 4, 5 )
    // CHECK-NEXT: crd[2] : ( 1, 2, 1, 2, 2 )
    // CHECK-NEXT: values : ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    // CHECK-NEXT: ----
    %tensora = tensor.empty() : tensor<5x4x3xf64, #TensorCSR>
    %tensor1 = tensor.insert %f1 into %tensora[%c3, %c0, %c1] : tensor<5x4x3xf64, #TensorCSR>
    %tensor2 = tensor.insert %f2 into %tensor1[%c3, %c0, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensor3 = tensor.insert %f3 into %tensor2[%c3, %c3, %c1] : tensor<5x4x3xf64, #TensorCSR>
    %tensor4 = tensor.insert %f4 into %tensor3[%c4, %c2, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensor5 = tensor.insert %f5 into %tensor4[%c4, %c3, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensorm = sparse_tensor.load %tensor5 hasInserts : tensor<5x4x3xf64, #TensorCSR>
    sparse_tensor.print %tensorm : tensor<5x4x3xf64, #TensorCSR>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 5, 4, 3 )
    // CHECK-NEXT: lvl = ( 5, 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 3, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 4 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 2, 3 )
    // CHECK-NEXT: values : ( 0, 1.1, 2.2, 0, 3.3, 0, 0, 0, 4.4, 0, 0, 5.5 )
    // CHECK-NEXT: ----
    %rowa = tensor.empty() : tensor<5x4x3xf64, #TensorRow>
    %row1 = tensor.insert %f1 into %rowa[%c3, %c0, %c1] : tensor<5x4x3xf64, #TensorRow>
    %row2 = tensor.insert %f2 into %row1[%c3, %c0, %c2] : tensor<5x4x3xf64, #TensorRow>
    %row3 = tensor.insert %f3 into %row2[%c3, %c3, %c1] : tensor<5x4x3xf64, #TensorRow>
    %row4 = tensor.insert %f4 into %row3[%c4, %c2, %c2] : tensor<5x4x3xf64, #TensorRow>
    %row5 = tensor.insert %f5 into %row4[%c4, %c3, %c2] : tensor<5x4x3xf64, #TensorRow>
    %rowm = sparse_tensor.load %row5 hasInserts : tensor<5x4x3xf64, #TensorRow>
    sparse_tensor.print %rowm : tensor<5x4x3xf64, #TensorRow>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 5, 4, 3 )
    // CHECK-NEXT: lvl = ( 5, 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 3, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 5 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 2, 3, 1, 2, 2, 3, 2 )
    // CHECK-NEXT: values : ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    // CHECK-NEXT: ----
    %ccoo = tensor.empty() : tensor<5x4x3xf64, #CCoo>
    %ccoo1 = tensor.insert %f1 into %ccoo[%c3, %c0, %c1] : tensor<5x4x3xf64, #CCoo>
    %ccoo2 = tensor.insert %f2 into %ccoo1[%c3, %c0, %c2] : tensor<5x4x3xf64, #CCoo>
    %ccoo3 = tensor.insert %f3 into %ccoo2[%c3, %c3, %c1] : tensor<5x4x3xf64, #CCoo>
    %ccoo4 = tensor.insert %f4 into %ccoo3[%c4, %c2, %c2] : tensor<5x4x3xf64, #CCoo>
    %ccoo5 = tensor.insert %f5 into %ccoo4[%c4, %c3, %c2] : tensor<5x4x3xf64, #CCoo>
    %ccoom = sparse_tensor.load %ccoo5 hasInserts : tensor<5x4x3xf64, #CCoo>
    sparse_tensor.print %ccoom : tensor<5x4x3xf64, #CCoo>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 5, 4, 3 )
    // CHECK-NEXT: lvl = ( 5, 4, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 0, 0, 0, 3, 5 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 2, 3, 1, 2, 2, 3, 2 )
    // CHECK-NEXT: values : ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    // CHECK-NEXT: ----
    %dcoo = tensor.empty() : tensor<5x4x3xf64, #DCoo>
    %dcoo1 = tensor.insert %f1 into %dcoo[%c3, %c0, %c1] : tensor<5x4x3xf64, #DCoo>
    %dcoo2 = tensor.insert %f2 into %dcoo1[%c3, %c0, %c2] : tensor<5x4x3xf64, #DCoo>
    %dcoo3 = tensor.insert %f3 into %dcoo2[%c3, %c3, %c1] : tensor<5x4x3xf64, #DCoo>
    %dcoo4 = tensor.insert %f4 into %dcoo3[%c4, %c2, %c2] : tensor<5x4x3xf64, #DCoo>
    %dcoo5 = tensor.insert %f5 into %dcoo4[%c4, %c3, %c2] : tensor<5x4x3xf64, #DCoo>
    %dcoom = sparse_tensor.load %dcoo5 hasInserts : tensor<5x4x3xf64, #DCoo>
    sparse_tensor.print %dcoom : tensor<5x4x3xf64, #DCoo>

    // Release resources.
    bufferization.dealloc_tensor %tensorm : tensor<5x4x3xf64, #TensorCSR>
    bufferization.dealloc_tensor %rowm : tensor<5x4x3xf64, #TensorRow>
    bufferization.dealloc_tensor %ccoom : tensor<5x4x3xf64, #CCoo>
    bufferization.dealloc_tensor %dcoom : tensor<5x4x3xf64, #DCoo>

    return
  }
}
