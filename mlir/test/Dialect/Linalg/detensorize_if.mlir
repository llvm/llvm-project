// RUN: mlir-opt %s -allow-unregistered-dialect -linalg-detensorize | FileCheck %s

#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func @main() -> (tensor<i32>) attributes {} {
  %c0 = constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<1xi32>
  %reshaped0 = linalg.tensor_reshape %0 [] : tensor<1xi32> into tensor<i32>
  %c10 = constant 10 : i32
  %1 = tensor.from_elements %c10 : tensor<1xi32>
  %reshaped1 = linalg.tensor_reshape %1 [] : tensor<1xi32> into tensor<i32>
  br ^bb1(%reshaped0 : tensor<i32>)

^bb1(%2: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %3 = linalg.init_tensor [] : tensor<i1>
  %4 = linalg.generic #attrs
    ins(%2, %reshaped1 : tensor<i32>, tensor<i32>)
    outs(%3 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  // no predecessors
      %8 = cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %5 = tensor.extract %4[] : tensor<i1>
  cond_br %5, ^bb2(%2 : tensor<i32>), ^bb3(%2 : tensor<i32>)

^bb2(%6: tensor<i32>):  // pred: ^bb1
  %7 = linalg.init_tensor [] : tensor<i32>
  %8 = linalg.generic #attrs
    ins(%6, %6 : tensor<i32>, tensor<i32>)
    outs(%7 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
      %9 = addi %arg0, %arg1 : i32
      linalg.yield %9 : i32
  } -> tensor<i32>
  br ^bb3(%8 : tensor<i32>)

^bb3(%10: tensor<i32>):  // pred: ^bb1
  return %10 : tensor<i32>
}

// CHECK-LABEL:  func @main()
// CHECK-NEXT:     constant 0
// CHECK-NEXT:     constant 10
// CHECK-NEXT:     br ^[[bb1:.*]](%{{.*}}: i32)
// CHECK-NEXT:   ^[[bb1]](%{{.*}}: i32):
// CHECK-NEXT:     tensor.from_elements %{{.*}}
// CHECK-NEXT:     linalg.tensor_reshape %{{.*}}
// CHECK-NEXT:     cmpi slt, %{{.*}}, %{{.*}}
// CHECK-NEXT:     cond_br %{{.*}}, ^[[bb2:.*]](%{{.*}} : tensor<i32>), ^bb3(%{{.*}} : tensor<i32>)
// CHECK-NEXT:   ^[[bb2]](%{{.*}}: tensor<i32>)
// CHECK-NEXT:     linalg.init_tensor
// CHECK-NEXT:     linalg.generic
// CHECK-NEXT:     ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32)
// CHECK-NEXT:       addi %{{.*}}, %{{.*}}
// CHECK-NEXT:       linalg.yield %{{.*}}
// CHECK-NEXT:     } -> tensor<i32>
// CHECK-NEXT:     br ^[[bb3:.*]](%{{.*}} : tensor<i32>)
// CHECK-NEXT:   ^[[bb3]](%{{.*}}: tensor<i32>)
// CHECK-NEXT:     return %{{.*}}
// CHECK-NEXT:   }
