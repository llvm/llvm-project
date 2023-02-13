// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//

// TODO: support slices on lib path
#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (1, 4, 1), (1, 4, 2) ]
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

  func.func @entry() {
    %c0 = arith.constant 0 : index
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

    // FIXME: investigate why a tensor copy is inserted for this slice
//    %dense = tensor.extract_slice %sa[1, 1][4, 4][1, 2] : tensor<8x8xf64> to
//                                                          tensor<4x4xf64>
//    %b = sparse_tensor.convert %dense : tensor<4x4xf64> to tensor<4x4xf64, #CSR>
//    // Foreach on sparse tensor instead of slice they should yield the same result.
//    //
//    // C_HECK-NEXT: 1
//    // C_HECK-NEXT: 0
//    // C_HECK-NEXT: 2.3
//    // C_HECK-NEXT: 2
//    // C_HECK-NEXT: 3
//    // C_HECK-NEXT: 1
//    // C_HECK-NEXT: 3
//    // C_HECK-NEXT: 2
//    // C_HECK-NEXT: 2.1
//    //
//    call @foreach_print_non_slice(%b) : (tensor<4x4xf64, #CSR>) -> ()
//    bufferization.dealloc_tensor %b : tensor<4x4xf64, #CSR>

    bufferization.dealloc_tensor %tmp : tensor<8x8xf64, #CSR>
    return
  }
}
