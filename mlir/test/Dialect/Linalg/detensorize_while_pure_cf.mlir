// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline="func.func(linalg-detensorize)" | FileCheck %s

#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func.func @main() -> () attributes {} {
  %c0 = arith.constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<1xi32>
  %reshaped0 = tensor.collapse_shape %0 [] : tensor<1xi32> into tensor<i32>
  %c10 = arith.constant 10 : i32
  %1 = tensor.from_elements %c10 : tensor<1xi32>
  %reshaped1 = tensor.collapse_shape %1 [] : tensor<1xi32> into tensor<i32>
  cf.br ^bb1(%reshaped0 : tensor<i32>)

^bb1(%2: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %3 = linalg.init_tensor [] : tensor<i1>
  %4 = linalg.generic #attrs
    ins(%2, %reshaped1 : tensor<i32>, tensor<i32>)
    outs(%3 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %5 = tensor.extract %4[] : tensor<i1>
  cf.cond_br %5, ^bb2(%2 : tensor<i32>), ^bb3

^bb2(%6: tensor<i32>):  // pred: ^bb1
  %7 = linalg.init_tensor [] : tensor<i32>
  %8 = linalg.generic #attrs
    ins(%6, %6 : tensor<i32>, tensor<i32>)
    outs(%7 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  
      %9 = arith.addi %arg0, %arg1 : i32
      linalg.yield %9 : i32
  } -> tensor<i32>
  cf.br ^bb1(%8 : tensor<i32>)

^bb3:  // pred: ^bb1
  return
}

// CHECK-LABEL: func @main
// CHECK-NEXT:    arith.constant 0 : i32
// CHECK-NEXT:    arith.constant 10
// CHECK-NEXT:    cf.br ^[[bb1:.*]](%{{.*}} : i32)
// CHECK-NEXT:  ^[[bb1]](%{{.*}}: i32)
// CHECK-NEXT:    %{{.*}} = arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK-NEXT:    cf.cond_br %{{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^[[bb3:.*]]
// CHECK-NEXT:  ^[[bb2]](%{{.*}}: i32)
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}}
// CHECK-NEXT:    cf.br ^[[bb1]](%{{.*}} : i32)
// CHECK-NEXT:  ^[[bb3]]:
// CHECK-NEXT:    return
// CHECK-NEXT:  }
