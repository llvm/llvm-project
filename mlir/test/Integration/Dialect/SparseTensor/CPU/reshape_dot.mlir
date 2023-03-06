// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = TENSOR0="%mlir_src_dir/test/Integration/data/test.mtx" \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

#COO_2D = #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ], posWidth = 32, crdWidth = 32 }>
#COO_3D = #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ], posWidth = 32, crdWidth = 32 }>

module {
  func.func private @printMemref3dF32(%ptr : tensor<?x?x?xf32>) attributes { llvm.emit_c_interface }
  func.func private @printMemref2dF32(%ptr : tensor<?x?xf32>) attributes { llvm.emit_c_interface }

  func.func @test_sparse_rhs(%arg0: tensor<5x6xf32>, %arg1: tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
    %0 = tensor.empty() : tensor<5x6xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32>, tensor<6x6xf32, #COO_2D>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
    %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>
    return %ret1 : tensor<?x?x?xf32>
  }

  func.func @test_sparse_all(%arg0: tensor<5x6xf32, #COO_2D>, %arg1: tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
    %0 = tensor.empty() : tensor<5x6xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32, #COO_2D>, tensor<6x6xf32, #COO_2D>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
    %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>
    return %ret1 : tensor<?x?x?xf32>
  }

  func.func @test_dense(%arg0: tensor<5x6xf32>, %arg1: tensor<6x2x3xf32>) -> tensor<?x?x?xf32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32> into tensor<6x6xf32>
    %0 = tensor.empty() : tensor<5x6xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32>, tensor<6x6xf32>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
    %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>
    return %ret1 :  tensor<?x?x?xf32>
  }


  func.func @entry() {
    // Setup two sparse vectors.
    %d1 = arith.constant sparse<
        [ [0, 0], [1, 1], [2, 2], [2, 3], [4, 5] ],
          [1.0,      2.0,    3.0,    4.0,   5.0]
    > : tensor<5x6xf32>

    %d2 = arith.constant sparse<
      [ [0, 0, 0], [1, 1, 1], [2, 1, 1] ],
        [     6.0,       7.0,      8.0]
    > : tensor<6x2x3xf32>

    %s1 = sparse_tensor.convert %d1 : tensor<5x6xf32> to tensor<5x6xf32, #COO_2D>
    %s2 = sparse_tensor.convert %d2 : tensor<6x2x3xf32> to tensor<6x2x3xf32, #COO_3D>

    //      CHECK: Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [5, 2, 3] strides = [6, 3, 1] data =
    // CHECK-NEXT:[
    // CHECK-SAME: [
    // CHECK-SAME:  [6,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    14,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    24,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]]]
    %do1 = call @test_dense(%d1, %d2) : (tensor<5x6xf32>, tensor<6x2x3xf32>) -> tensor<?x?x?xf32>
    call @printMemref3dF32(%do1) : (tensor<?x?x?xf32>) -> ()

    // Same results.
    // CHECK-NEXT: Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [5, 2, 3] strides = [6, 3, 1] data =
    // CHECK-NEXT:[
    // CHECK-SAME: [
    // CHECK-SAME:  [6,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    14,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    24,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]]]
    %so1 = call @test_sparse_rhs(%d1, %s2): (tensor<5x6xf32>, tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32>
    call @printMemref3dF32(%so1) : (tensor<?x?x?xf32>) -> ()

    // Same results.
    // CHECK-NEXT: Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [5, 2, 3] strides = [6, 3, 1] data =
    // CHECK-NEXT:[
    // CHECK-SAME: [
    // CHECK-SAME:  [6,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    14,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    24,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]],
    // CHECK-NEXT: [
    // CHECK-SAME:  [0,    0,    0],
    // CHECK-NEXT:  [0,    0,    0]]]
    %so2 = call @test_sparse_all(%s1, %s2): (tensor<5x6xf32, #COO_2D>, tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32>
    call @printMemref3dF32(%so2) : (tensor<?x?x?xf32>) -> ()

    bufferization.dealloc_tensor %s1 : tensor<5x6xf32, #COO_2D>
    bufferization.dealloc_tensor %s2 : tensor<6x2x3xf32, #COO_3D>
    bufferization.dealloc_tensor %do1 : tensor<?x?x?xf32>
    bufferization.dealloc_tensor %so1 : tensor<?x?x?xf32>
    bufferization.dealloc_tensor %so2 : tensor<?x?x?xf32>
    return
  }
}
