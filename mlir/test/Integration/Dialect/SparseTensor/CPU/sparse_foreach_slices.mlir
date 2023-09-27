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

// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false

// RUN: %{compile} | %{run} | FileCheck %s

// TODO: support slices on lib path

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSR_SLICE = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(1, 4, 1)>, d1 : #sparse_tensor<slice(1, 4, 2)>) -> (d0 : dense, d1 : compressed)
}>

#CSR_SLICE_DYN = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(?, ?, ?)>, d1 : #sparse_tensor<slice(?, ?, ?)>) -> (d0 : dense, d1 : compressed)
}>

#COO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#COO_SLICE = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(1, 4, 1)>, d1 : #sparse_tensor<slice(1, 4, 2)>) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#COO_SLICE_DYN = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(?, ?, ?)>, d1 : #sparse_tensor<slice(?, ?, ?)>) -> (d0 : compressed(nonunique), d1 : singleton)
}>



module {
  func.func @foreach_print_non_slice(%A: tensor<4x4xf64, #CSR>) {
    sparse_tensor.foreach in %A : tensor<4x4xf64, #CSR> do {
    ^bb0(%1: index, %2: index, %v: f64) :
      vector.print %1: index
      vector.print %2: index
      vector.print %v: f64
    }
    return
  }

  func.func @foreach_print_slice(%A: tensor<4x4xf64, #CSR_SLICE>) {
    sparse_tensor.foreach in %A : tensor<4x4xf64, #CSR_SLICE> do {
    ^bb0(%1: index, %2: index, %v: f64) :
      vector.print %1: index
      vector.print %2: index
      vector.print %v: f64
    }
    return
  }

  func.func @foreach_print_slice_dyn(%A: tensor<?x?xf64, #CSR_SLICE_DYN>) {
    sparse_tensor.foreach in %A : tensor<?x?xf64, #CSR_SLICE_DYN> do {
    ^bb0(%1: index, %2: index, %v: f64) :
      vector.print %1: index
      vector.print %2: index
      vector.print %v: f64
    }
    return
  }

  func.func @foreach_print_slice_coo(%A: tensor<4x4xf64, #COO_SLICE>) {
    sparse_tensor.foreach in %A : tensor<4x4xf64, #COO_SLICE> do {
    ^bb0(%1: index, %2: index, %v: f64) :
      vector.print %1: index
      vector.print %2: index
      vector.print %v: f64
    }
    return
  }

  func.func @foreach_print_slice_coo_dyn(%A: tensor<?x?xf64, #COO_SLICE_DYN>) {
    sparse_tensor.foreach in %A : tensor<?x?xf64, #COO_SLICE_DYN> do {
    ^bb0(%1: index, %2: index, %v: f64) :
      vector.print %1: index
      vector.print %2: index
      vector.print %v: f64
    }
    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %sa = arith.constant dense<[
        [ 0.0, 2.1, 0.0, 0.0, 0.0, 6.1, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
        [ 0.0, 0.0, 0.1, 0.0, 0.0, 2.1, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 3.1, 0.0, 0.0, 0.0 ],
        [ 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 3.3, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
    ]> : tensor<8x8xf64>


    %tmp = sparse_tensor.convert %sa : tensor<8x8xf64> to tensor<8x8xf64, #CSR>
    %a = tensor.extract_slice %tmp[1, 1][4, 4][1, 2] : tensor<8x8xf64, #CSR> to
                                                       tensor<4x4xf64, #CSR_SLICE>

    %tmp_coo = sparse_tensor.convert %sa : tensor<8x8xf64> to tensor<8x8xf64, #COO>
    %a_coo = tensor.extract_slice %tmp_coo[1, 1][4, 4][1, 2] : tensor<8x8xf64, #COO> to
                                                               tensor<4x4xf64, #COO_SLICE>
    // Foreach on sparse tensor slices directly
    //
    // CHECK: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 2.3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 1
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 2.1
    //
    call @foreach_print_slice(%a) : (tensor<4x4xf64, #CSR_SLICE>) -> ()
    // Same results for COO
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 2.3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 1
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 2.1
    //
    call @foreach_print_slice_coo(%a_coo) : (tensor<4x4xf64, #COO_SLICE>) -> ()

    %dense = tensor.extract_slice %sa[1, 1][4, 4][1, 2] : tensor<8x8xf64> to
                                                          tensor<4x4xf64>
    %b = sparse_tensor.convert %dense : tensor<4x4xf64> to tensor<4x4xf64, #CSR>
    // Foreach on sparse tensor instead of slice they should yield the same result.
    //
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 2.3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 1
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 2.1
    //
    call @foreach_print_non_slice(%b) : (tensor<4x4xf64, #CSR>) -> ()

    // The same slice, but with dynamic encoding.
    // TODO: Investigates why reusing the same %tmp above would cause bufferization
    // errors.
    %tmp1 = sparse_tensor.convert %sa : tensor<8x8xf64> to tensor<8x8xf64, #CSR>
    %a_dyn = tensor.extract_slice %tmp1[%c1, %c1][%c4, %c4][%c1, %c2] : tensor<8x8xf64, #CSR> to
                                                                        tensor<?x?xf64, #CSR_SLICE_DYN>

    %tmp1_coo = sparse_tensor.convert %sa : tensor<8x8xf64> to tensor<8x8xf64, #COO>
    %a_dyn_coo = tensor.extract_slice %tmp1_coo[%c1, %c1][%c4, %c4][%c1, %c2] : tensor<8x8xf64, #COO> to
                                                                                tensor<?x?xf64, #COO_SLICE_DYN>
    //
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 2.3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 1
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 2.1
    //
    call @foreach_print_slice_dyn(%a_dyn) : (tensor<?x?xf64, #CSR_SLICE_DYN>) -> ()
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 2.3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 1
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-NEXT: 2.1
    //
    call @foreach_print_slice_coo_dyn(%a_dyn_coo) : (tensor<?x?xf64, #COO_SLICE_DYN>) -> ()

    bufferization.dealloc_tensor %tmp : tensor<8x8xf64, #CSR>
    bufferization.dealloc_tensor %tmp1 : tensor<8x8xf64, #CSR>
    bufferization.dealloc_tensor %tmp_coo : tensor<8x8xf64, #COO>
    bufferization.dealloc_tensor %tmp1_coo : tensor<8x8xf64, #COO>
    bufferization.dealloc_tensor %b : tensor<4x4xf64, #CSR>
    return
  }
}
