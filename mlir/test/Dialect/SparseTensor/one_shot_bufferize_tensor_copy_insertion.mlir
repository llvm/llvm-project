// RUN: mlir-opt %s -test-tensor-copy-insertion | FileCheck %s
// RUN: mlir-opt %s -test-tensor-copy-insertion="bufferize-function-boundaries" | FileCheck %s --check-prefix=CHECK-FUNC

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL: func @bufferization_alloc_tensor
// CHECK-FUNC-LABEL: func @bufferization_alloc_tensor
func.func @bufferization_alloc_tensor() -> tensor<20x40xf32, #DCSR> {
  // CHECK: bufferization.alloc_tensor()
  // CHECK-FUNC: bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  %1 = sparse_tensor.load %0 : tensor<20x40xf32, #DCSR>
  return %1 : tensor<20x40xf32, #DCSR>
}

!Filename = !llvm.ptr
// CHECK-LABEL: func @sparse_tensor_new
// CHECK-FUNC-LABEL: func @sparse_tensor_new
func.func @sparse_tensor_new(%file: !Filename) -> tensor<20x40xf32, #DCSR> {
  // CHECK: sparse_tensor.new {{.*}}
  // CHECK-FUNC: sparse_tensor.new {{.*}}
  %0 = sparse_tensor.new %file : !Filename to tensor<20x40xf32, #DCSR>
  return %0 : tensor<20x40xf32, #DCSR>
}

// CHECK-LABEL: func @sparse_tensor_convert
// CHECK-FUNC-LABEL: func @sparse_tensor_convert
func.func @sparse_tensor_convert() -> tensor<20x40xf32> {
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  // CHECK-FUNC: %[[alloc:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  // CHECK: %[[loaded:.*]] = sparse_tensor.load %[[alloc]]
  // CHECK-FUNC: %[[loaded:.*]] = sparse_tensor.load %[[alloc]]
  %1 = sparse_tensor.load %0 : tensor<20x40xf32, #DCSR>
  // CHECK: sparse_tensor.convert %[[loaded]]
  // CHECK-FUNC: sparse_tensor.convert %[[loaded]]
  %2 = sparse_tensor.convert %1 : tensor<20x40xf32, #DCSR> to tensor<20x40xf32>
  return %2 : tensor<20x40xf32>
}

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

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
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[argb]]) : tensor<10xf32>
  // CHECK: linalg.generic {{.*}} outs(%[[alloc]]
  // CHECK-FUNC: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[argb]]) : tensor<10xf32>
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

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>

// linalg.generic with sparse tensors does not necessarily bufferize to
// element-wise access into the underlying sparse data structures.

// CHECK-LABEL: func @sparse_non_elementwise(
func.func @sparse_non_elementwise(%arg0: tensor<64x64xf32, #sparse>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[alloc0:.*]] = bufferization.alloc_tensor()
  // CHECK: %[[alloc1:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<64x64xf32>
  // CHECK: %[[generic0:.*]] = linalg.generic {{.*}} outs(%[[alloc1]] : {{.*}})
  %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<64x64xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  } -> tensor<64x64xf32>
  // CHECK: linalg.generic {{.*}} outs(%[[generic0]] : {{.*}})
  %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg2, %arg2 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.mulf %in, %in_0 : f32
    %5 = arith.addf %out, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<64x64xf32>
  // CHECK: linalg.generic {{.*}} outs(%[[alloc0]] : {{.*}})
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<64x64xf32, #sparse>, tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) attrs =  {sorted = true} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.mulf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<64x64xf32>
  return %3 : tensor<64x64xf32>
}
