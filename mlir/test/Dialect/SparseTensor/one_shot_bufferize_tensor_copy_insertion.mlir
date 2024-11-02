// RUN: mlir-opt %s -test-tensor-copy-insertion="allow-return-allocs" | FileCheck %s
// RUN: mlir-opt %s -test-tensor-copy-insertion="bufferize-function-boundaries allow-return-allocs" | FileCheck %s --check-prefix=CHECK-FUNC

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

// CHECK-LABEL: func @bufferization_alloc_tensor
// CHECK-FUNC-LABEL: func @bufferization_alloc_tensor
func.func @bufferization_alloc_tensor() -> tensor<20x40xf32, #DCSR> {
  // CHECK: bufferization.alloc_tensor() {bufferization.escape = [false]}
  // CHECK-FUNC: bufferization.alloc_tensor() {bufferization.escape = [true]}
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  %1 = sparse_tensor.load %0 : tensor<20x40xf32, #DCSR>
  return %1 : tensor<20x40xf32, #DCSR>
}

!Filename = !llvm.ptr<i8>
// CHECK-LABEL: func @sparse_tensor_new
// CHECK-FUNC-LABEL: func @sparse_tensor_new
func.func @sparse_tensor_new(%file: !Filename) -> tensor<20x40xf32, #DCSR> {
  // CHECK: sparse_tensor.new {{.*}} {bufferization.escape = [false]}
  // CHECK-FUNC: sparse_tensor.new {{.*}} {bufferization.escape = [true]}
  %0 = sparse_tensor.new %file : !Filename to tensor<20x40xf32, #DCSR>
  return %0 : tensor<20x40xf32, #DCSR>
}

// CHECK-LABEL: func @sparse_tensor_convert
// CHECK-FUNC-LABEL: func @sparse_tensor_convert
func.func @sparse_tensor_convert() -> tensor<20x40xf32> {
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() {bufferization.escape = [false]}
  // CHECK-FUNC: %[[alloc:.*]] = bufferization.alloc_tensor() {bufferization.escape = [false]}
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  // CHECK: %[[loaded:.*]] = sparse_tensor.load %[[alloc]]
  // CHECK-FUNC: %[[loaded:.*]] = sparse_tensor.load %[[alloc]]
  %1 = sparse_tensor.load %0 : tensor<20x40xf32, #DCSR>
  // CHECK: sparse_tensor.convert %[[loaded]] {bufferization.escape = [false]}
  // CHECK-FUNC: sparse_tensor.convert %[[loaded]] {bufferization.escape = [true]}
  %2 = sparse_tensor.convert %1 : tensor<20x40xf32, #DCSR> to tensor<20x40xf32>
  return %2 : tensor<20x40xf32>
}

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A (in)
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @update_notinplace(
//  CHECK-SAME:    %[[argb:.*]]: tensor<10xf32>
// CHECK-FUNC-LABEL: func @update_notinplace(
//  CHECK-FUNC-SAME:    %[[argb:.*]]: tensor<10xf32>
func.func @update_notinplace(%argb: tensor<10xf32>, %arga: tensor<10xf32, #SV>)
  -> (tensor<10xf32>, tensor<10xf32>)
{
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[argb]]) {bufferization.escape = [false]} : tensor<10xf32>
  // CHECK: linalg.generic {{.*}} outs(%[[alloc]]
  // CHECK-FUNC: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[argb]]) {bufferization.escape = [true]} : tensor<10xf32>
  // CHECK-FUNC: linalg.generic {{.*}} outs(%[[alloc]]
  %0 = linalg.generic #trait
  ins(%arga: tensor<10xf32, #SV>)
  outs(%argb: tensor<10xf32>) {
    ^bb(%a: f32, %x : f32):
      %up = arith.addf %a, %x : f32
      linalg.yield %up : f32
  } -> tensor<10xf32>
  return %0, %argb : tensor<10xf32>, tensor<10xf32>
}
