// RUN: mlir-query %s -c "m anyOf(getAllDefinitions(hasOpName(\"arith.addf\"),2),getAllDefinitions(hasOpName(\"tensor.extract\"),1))" | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @slice_use_from_above(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c2 = arith.constant 2 : index
    %extracted = tensor.extract %collapsed[%c2] : tensor<25xf32>
    %2 = arith.addf %extracted, %extracted : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  return
}

// CHECK: Match #1:
// CHECK: %[[LINALG:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} 
// CHECK-SAME: ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>)

// CHECK: {{.*}}.mlir:7:10: note: "root" binds here
// CHECK: %[[ADDF1:.*]] = arith.addf %in, %in : f32

// CHECK: Match #2:
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[LINALG]] {{\[\[.*\]\]}} : tensor<5x5xf32> into tensor<25xf32>
// CHECK: %[[C2:.*]] = arith.constant {{.*}} : index

// CHECK: {{.*}}.mlir:14:18: note: "root" binds here
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[COLLAPSED]][%[[C2]]] : tensor<25xf32>

// CHECK: Match #3:
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[LINALG]] {{\[\[.*\]\]}} : tensor<5x5xf32> into tensor<25xf32>
// CHECK: %[[C2:.*]] = arith.constant {{.*}} : index
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[COLLAPSED]][%[[C2]]] : tensor<25xf32>

// CHECK: {{.*}}.mlir:15:10: note: "root" binds here
// CHECK: %[[ADDF2:.*]] = arith.addf %[[EXTRACTED]], %[[EXTRACTED]] : f32  
