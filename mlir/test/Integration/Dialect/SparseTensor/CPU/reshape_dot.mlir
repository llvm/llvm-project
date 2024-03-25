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

// RUN: %{compile} | %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{env} %{run_sve} | FileCheck %s %}


#COO_2D = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton), posWidth = 32, crdWidth = 32 }>
#COO_3D = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton), posWidth = 32, crdWidth = 32 }>

module {
  func.func private @printMemref3dF32(%ptr : tensor<?x?x?xf32> {bufferization.access = "read"}) attributes { llvm.emit_c_interface }
  func.func private @printMemref2dF32(%ptr : tensor<?x?xf32> {bufferization.access = "read"}) attributes { llvm.emit_c_interface }

  func.func @test_sparse_rhs(%arg0: tensor<5x6xf32>, %arg1: tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
    %0 = tensor.empty() : tensor<5x6xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32>, tensor<6x6xf32, #COO_2D>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
    %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>

    // Note: tensor.collapse_shape is a metadata-only operation on dense tensors
    // but requires reallocation on sparse tensors.
    bufferization.dealloc_tensor %collapsed : tensor<6x6xf32, #COO_2D>

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

    // Note: tensor.collapse_shape is a metadata-only operation on dense tensors
    // but requires reallocation on sparse tensors.
    bufferization.dealloc_tensor %collapsed : tensor<6x6xf32, #COO_2D>

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

  func.func @test_sparse_all_2(%arg0: tensor<5x6xf32, #COO_2D>, %arg1: tensor<2x3x6xf32, #COO_3D>) -> tensor<?x?x?xf32> {
    // collapse the first two level this time, as this is the level requires coiterations.
    %collapsed = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<2x3x6xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
    %0 = tensor.empty() : tensor<5x6xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32, #COO_2D>, tensor<6x6xf32, #COO_2D>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
    %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>

    // Note: tensor.collapse_shape is a metadata-only operation on dense tensors
    // but requires reallocation on sparse tensors.
    bufferization.dealloc_tensor %collapsed : tensor<6x6xf32, #COO_2D>

    return %ret1 : tensor<?x?x?xf32>
  }


  func.func @main() {
    // Setup two sparse vectors.
    %d1 = arith.constant sparse<
        [ [0, 0], [1, 1], [2, 2], [2, 3], [4, 5] ],
          [1.0,      2.0,    3.0,    4.0,   5.0]
    > : tensor<5x6xf32>

    %d2 = arith.constant sparse<
      [ [0, 0, 0], [1, 1, 1], [2, 1, 1] ],
        [     6.0,       7.0,      8.0]
    > : tensor<6x2x3xf32>
    %shape = arith.constant dense<[2, 3, 6]> : tensor<3xi32>

    %d3 = tensor.reshape %d2(%shape): (tensor<6x2x3xf32>, tensor<3xi32>) -> tensor<2x3x6xf32>
    %s1 = sparse_tensor.convert %d1 : tensor<5x6xf32> to tensor<5x6xf32, #COO_2D>
    %s2 = sparse_tensor.convert %d2 : tensor<6x2x3xf32> to tensor<6x2x3xf32, #COO_3D>
    %s3 = sparse_tensor.convert %d3 : tensor<2x3x6xf32> to tensor<2x3x6xf32, #COO_3D>

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
    %so3 = call @test_sparse_all_2(%s1, %s3): (tensor<5x6xf32, #COO_2D>, tensor<2x3x6xf32, #COO_3D>) -> tensor<?x?x?xf32>
    call @printMemref3dF32(%so2) : (tensor<?x?x?xf32>) -> ()

    bufferization.dealloc_tensor %s1 : tensor<5x6xf32, #COO_2D>
    bufferization.dealloc_tensor %s2 : tensor<6x2x3xf32, #COO_3D>
    bufferization.dealloc_tensor %s3 : tensor<2x3x6xf32, #COO_3D>
    bufferization.dealloc_tensor %do1 : tensor<?x?x?xf32>
    bufferization.dealloc_tensor %so1 : tensor<?x?x?xf32>
    bufferization.dealloc_tensor %so2 : tensor<?x?x?xf32>
    bufferization.dealloc_tensor %so3 : tensor<?x?x?xf32>

    return
  }
}
