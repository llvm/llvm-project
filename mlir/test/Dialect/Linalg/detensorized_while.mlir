// RUN: mlir-opt %s -linalg-detensorize | FileCheck %s

#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func @main(%farg0: tensor<i32>, %farg1: tensor<i32>) -> tensor<i32> attributes {} {
  br ^bb1(%farg0 : tensor<i32>)

^bb1(%0: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %1 = linalg.init_tensor [] : tensor<i1>
  %2 = linalg.generic #attrs
    ins(%0, %farg1 : tensor<i32>, tensor<i32>)
    outs(%1 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  // no predecessors
      %8 = cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %3 = tensor.extract %2[] : tensor<i1>
  cond_br %3, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)

^bb2(%4: tensor<i32>):  // pred: ^bb1
  %5 = linalg.init_tensor [] : tensor<i32>
  %6 = linalg.generic #attrs
    ins(%4, %4 : tensor<i32>, tensor<i32>)
    outs(%5 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
      %8 = addi %arg0, %arg1 : i32
      linalg.yield %8 : i32
  } -> tensor<i32>
  br ^bb1(%6 : tensor<i32>)

^bb3(%7: tensor<i32>):  // pred: ^bb1
  return %7 : tensor<i32>
}

// CHECK-LABEL: func @main
// CHECK-SAME:    (%{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>)
// CHECK:         tensor.extract {{.*}}
// CHECK:         br ^[[bb1:.*]](%{{.*}} : i32)
// CHECK:       ^[[bb1]](%{{.*}}: i32)
// CHECK:         cmpi slt, {{.*}}
// CHECK:         cond_br {{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^[[bb3:.*]](%{{.*}} : i32)
// CHECK:       ^[[bb2]](%{{.*}}: i32)
// CHECK:         addi {{.*}}
// CHECK:         br ^[[bb1]](%{{.*}} : i32)
// CHECK:       ^[[bb3]](%{{.*}}: i32)
// CHECK:         tensor.from_elements {{.*}}
// CHECK:         linalg.tensor_reshape {{.*}}
// CHECK:         return %{{.*}} : tensor<i32>
