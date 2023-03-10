// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//

// TODO: support lib path.

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#DCSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  slice = [ (0, 4, 1), (0, 8, 1) ]
}>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (0, 4, 1), (0, 8, 1) ]
}>

#CSR_SLICE_1 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (0, 4, 2), (0, 4, 1) ]
}>

#DCSR_SLICE_1 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  slice = [ (0, 4, 2), (1, 4, 1) ]
}>

#CSR_SLICE_dyn = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (?, 4, ?), (?, 4, ?) ]
}>

#DCSR_SLICE_dyn = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  slice = [ (?, 4, ?), (?, 4, ?) ]
}>


module {
  func.func private @printMemrefF64(%ptr : tensor<*xf64>)
  func.func private @printMemref1dF64(%ptr : memref<?xf64>) attributes { llvm.emit_c_interface }

  //
  // Computes C = A x B with all matrices dynamic sparse slice (SpMSpM) in CSR and DCSR
  //
  func.func @matmul_dyn(%A: tensor<4x4xf64, #CSR_SLICE_dyn>,
                        %B: tensor<4x4xf64, #DCSR_SLICE_dyn>) -> tensor<4x4xf64, #CSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #CSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x4xf64, #CSR_SLICE_dyn>, tensor<4x4xf64, #DCSR_SLICE_dyn>)
         outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    return %D: tensor<4x4xf64, #CSR>
  }

  //
  // Computes C = A x B with one matrix CSR sparse slices and the other DSCR sparse slice.
  //
  func.func @matmul1(%A: tensor<4x4xf64, #CSR_SLICE_1>,
                     %B: tensor<4x4xf64, #DCSR_SLICE_1>) -> tensor<4x4xf64, #CSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #CSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x4xf64, #CSR_SLICE_1>, tensor<4x4xf64, #DCSR_SLICE_1>)
         outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    return %D: tensor<4x4xf64, #CSR>
  }

  //
  // Computes C = A x B with one matrix CSR sparse slice and the other CSR sparse tensor.
  //
  func.func @matmul2(%A: tensor<4x8xf64, #CSR_SLICE>,
                     %B: tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #CSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #CSR_SLICE>, tensor<8x4xf64, #CSR>)
         outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
    return %D: tensor<4x4xf64, #CSR>
  }

  //
  // Computes C = A x B with one matrix DCSR sparse slice and the other DCSR sparse tensor.
  //
  func.func @matmul3(%A: tensor<4x8xf64, #DCSR_SLICE>,
                     %B: tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
    %C = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
    %D = linalg.matmul
      ins(%A, %B: tensor<4x8xf64, #DCSR_SLICE>, tensor<8x4xf64, #DCSR>)
         outs(%C: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
    return %D: tensor<4x4xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c_0 = arith.constant 0 : index
    %c_1 = arith.constant 1 : index
    %c_2 = arith.constant 2 : index
    %f0 = arith.constant 0.0 : f64

    %sa = arith.constant dense<[
        [ 0.0, 2.1, 0.0, 0.0, 0.0, 6.1, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
        [ 0.0, 2.1, 0.0, 0.0, 0.0, 6.1, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
    ]> : tensor<8x8xf64>
    %sb = arith.constant dense<[
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 0.0, 0.0, 2.0, 0.0 ],
        [ 0.0, 3.0, 0.0, 0.0 ],
        [ 4.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 5.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 6.0, 0.0 ],
        [ 0.0, 0.0, 7.0, 8.0 ]
    ]> : tensor<8x4xf64>
    %zero = arith.constant dense<0.0> : tensor<4x4xf64>

    // Convert all these matrices to sparse format.
    %tmp = sparse_tensor.convert %sa : tensor<8x8xf64> to tensor<8x8xf64, #DCSR>
    %a = tensor.extract_slice %tmp[0, 0][4, 8][1, 1] : tensor<8x8xf64, #DCSR> to tensor<4x8xf64, #DCSR_SLICE>
    %b = sparse_tensor.convert %sb : tensor<8x4xf64> to tensor<8x4xf64, #DCSR>

    %2 = call @matmul3(%a, %b)
       : (tensor<4x8xf64, #DCSR_SLICE>,
          tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>

    // DCSR test
    //
    // CHECK:       [0,   30.5,   4.2,   0],
    // CHECK-NEXT:  [0,   0,   0,   0],
    // CHECK-NEXT:  [0,   0,   4.6,   0],
    // CHECK-NEXT:  [0,   0,   7,   8]
    //
    %c2 = sparse_tensor.convert %2 : tensor<4x4xf64, #DCSR> to tensor<4x4xf64>
    %c2u = tensor.cast %c2 : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%c2u) : (tensor<*xf64>) -> ()

    %t1 = sparse_tensor.convert %sa : tensor<8x8xf64> to tensor<8x8xf64, #CSR>
    %a1 = tensor.extract_slice %t1[0, 0][4, 8][1, 1] : tensor<8x8xf64, #CSR> to tensor<4x8xf64, #CSR_SLICE>
    %b1 = sparse_tensor.convert %sb : tensor<8x4xf64> to tensor<8x4xf64, #CSR>
    %3 = call @matmul2(%a1, %b1)
       : (tensor<4x8xf64, #CSR_SLICE>,
          tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>

    // CSR test
    //
    // CHECK:       [0,   30.5,   4.2,   0],
    // CHECK-NEXT:  [0,   0,   0,   0],
    // CHECK-NEXT:  [0,   0,   4.6,   0],
    // CHECK-NEXT:  [0,   0,   7,   8]
    //
    %c3 = sparse_tensor.convert %3 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %c3u = tensor.cast %c3 : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%c3u) : (tensor<*xf64>) -> ()

    // slice x slice
    //
    // CHECK:      [2.3,   0,   0,   0],
    // CHECK-NEXT: [6.9,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0],
    // CHECK-NEXT: [12.6,   0,   0,   0]]
    //
    %s1 = tensor.extract_slice %tmp[0, 1][4, 4][2, 1] : tensor<8x8xf64, #DCSR> to tensor<4x4xf64, #DCSR_SLICE_1>
    %s2 = tensor.extract_slice %b1[0, 0][4, 4][2, 1] : tensor<8x4xf64, #CSR> to tensor<4x4xf64, #CSR_SLICE_1>
    %4 = call @matmul1(%s2, %s1)
       : (tensor<4x4xf64, #CSR_SLICE_1>,
          tensor<4x4xf64, #DCSR_SLICE_1>) -> tensor<4x4xf64, #CSR>
    %c4 = sparse_tensor.convert %4 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %c4u = tensor.cast %c4 : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%c4u) : (tensor<*xf64>) -> ()

    // slice x slice (same as above, but with dynamic stride information)
    //
    // CHECK:      [2.3,   0,   0,   0],
    // CHECK-NEXT: [6.9,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0],
    // CHECK-NEXT: [12.6,   0,   0,   0]]
    //
    %s1_dyn = tensor.extract_slice %tmp[%c_0, %c_1][4, 4][%c_2, %c_1] : tensor<8x8xf64, #DCSR> to tensor<4x4xf64, #DCSR_SLICE_dyn>
    %s2_dyn = tensor.extract_slice %b1[%c_0, %c_0][4, 4][%c_2, %c_1] : tensor<8x4xf64, #CSR> to tensor<4x4xf64, #CSR_SLICE_dyn>
    %dyn_4 = call @matmul_dyn(%s2_dyn, %s1_dyn)
       : (tensor<4x4xf64, #CSR_SLICE_dyn>,
          tensor<4x4xf64, #DCSR_SLICE_dyn>) -> tensor<4x4xf64, #CSR>

    %c4_dyn = sparse_tensor.convert %dyn_4 : tensor<4x4xf64, #CSR> to tensor<4x4xf64>
    %c4u_dyn = tensor.cast %c4_dyn : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%c4u_dyn) : (tensor<*xf64>) -> ()

    // sparse slices should generate the same result as dense slices
    //
    // CHECK:      [2.3,   0,   0,   0],
    // CHECK-NEXT: [6.9,   0,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0],
    // CHECK-NEXT: [12.6,   0,   0,   0]]
    //
    %ds1 = tensor.extract_slice %sa[0, 1][4, 4][2, 1] : tensor<8x8xf64> to tensor<4x4xf64>
    %ds2 = tensor.extract_slice %sb[0, 0][4, 4][2, 1] : tensor<8x4xf64> to tensor<4x4xf64>

    %d = bufferization.alloc_tensor() copy(%zero) : tensor<4x4xf64>
    %r = linalg.matmul ins(%ds2, %ds1: tensor<4x4xf64>, tensor<4x4xf64>)
                       outs(%d: tensor<4x4xf64>) -> tensor<4x4xf64>
    %du = tensor.cast %r : tensor<4x4xf64> to tensor<*xf64>
    call @printMemrefF64(%du) : (tensor<*xf64>) -> ()

    // Releases resources (we do not need to deallocate slices).
    bufferization.dealloc_tensor %b1 : tensor<8x4xf64, #CSR>
    bufferization.dealloc_tensor %t1 : tensor<8x8xf64, #CSR>
    bufferization.dealloc_tensor %b  : tensor<8x4xf64, #DCSR>
    bufferization.dealloc_tensor %tmp: tensor<8x8xf64, #DCSR>
    bufferization.dealloc_tensor %4  : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %3  : tensor<4x4xf64, #CSR>
    bufferization.dealloc_tensor %2  : tensor<4x4xf64, #DCSR>
    bufferization.dealloc_tensor %dyn_4 : tensor<4x4xf64, #CSR>

    return
  }
}
