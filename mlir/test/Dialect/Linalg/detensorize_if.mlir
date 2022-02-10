// RUN: mlir-opt %s -split-input-file -allow-unregistered-dialect -pass-pipeline="builtin.func(linalg-detensorize)" | FileCheck %s

#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func @main() -> (tensor<i32>) attributes {} {
  %c0 = arith.constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<i32>
  %c10 = arith.constant 10 : i32
  %1 = tensor.from_elements %c10 : tensor<i32>
  cf.br ^bb1(%0 : tensor<i32>)

^bb1(%2: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %3 = linalg.init_tensor [] : tensor<i1>
  %4 = linalg.generic #attrs
    ins(%2, %1 : tensor<i32>, tensor<i32>)
    outs(%3 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %5 = tensor.extract %4[] : tensor<i1>
  cf.cond_br %5, ^bb2(%2 : tensor<i32>), ^bb3(%2 : tensor<i32>)

^bb2(%6: tensor<i32>):  // pred: ^bb1
  %7 = linalg.init_tensor [] : tensor<i32>
  %8 = linalg.generic #attrs
    ins(%6, %6 : tensor<i32>, tensor<i32>)
    outs(%7 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  
      %9 = arith.addi %arg0, %arg1 : i32
      linalg.yield %9 : i32
  } -> tensor<i32>
  cf.br ^bb3(%8 : tensor<i32>)

^bb3(%10: tensor<i32>):  // pred: ^bb1
  return %10 : tensor<i32>
}

// CHECK-LABEL:  func @main()
// CHECK-NEXT:     arith.constant 0
// CHECK-NEXT:     arith.constant 10
// CHECK-NEXT:     cf.br ^[[bb1:.*]](%{{.*}}: i32)
// CHECK-NEXT:   ^[[bb1]](%{{.*}}: i32):
// CHECK-NEXT:     arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK-NEXT:     cf.cond_br %{{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^bb3(%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb2]](%{{.*}}: i32)
// CHECK-NEXT:     arith.addi %{{.*}}, %{{.*}}
// CHECK-NEXT:     cf.br ^[[bb3:.*]](%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb3]](%{{.*}}: i32)
// CHECK-NEXT:     tensor.from_elements %{{.*}} : tensor<i32>
// CHECK-NEXT:     return %{{.*}}
// CHECK-NEXT:   }

// -----

// Similar to the above test with one change: one of the block after the
// if-condition passes/forwards its tensor argument to another block.

#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func @main() -> (tensor<i32>) attributes {} {
  %c0 = arith.constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<i32>
  %c10 = arith.constant 10 : i32
  %1 = tensor.from_elements %c10 : tensor<i32>
  cf.br ^bb1(%0 : tensor<i32>)

^bb1(%2: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %3 = linalg.init_tensor [] : tensor<i1>
  %4 = linalg.generic #attrs
    ins(%2, %1 : tensor<i32>, tensor<i32>)
    outs(%3 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %5 = tensor.extract %4[] : tensor<i1>
  cf.cond_br %5, ^bb2(%2 : tensor<i32>), ^bb3(%2 : tensor<i32>)

^bb2(%6: tensor<i32>):  // pred: ^bb1
  %7 = linalg.init_tensor [] : tensor<i32>
  %8 = linalg.generic #attrs
    ins(%6, %6 : tensor<i32>, tensor<i32>)
    outs(%7 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  
      %9 = arith.addi %arg0, %arg1 : i32
      linalg.yield %9 : i32
  } -> tensor<i32>
  cf.br ^bb3(%8 : tensor<i32>)

^bb3(%10: tensor<i32>):  // pred: ^bb1
  cf.br ^bb4(%10 : tensor<i32>)

^bb4(%11: tensor<i32>):  // pred: ^bb1
  return %11 : tensor<i32>
}

// CHECK-LABEL:  func @main()
// CHECK-NEXT:     arith.constant 0
// CHECK-NEXT:     arith.constant 10
// CHECK-NEXT:     cf.br ^[[bb1:.*]](%{{.*}}: i32)
// CHECK-NEXT:   ^[[bb1]](%{{.*}}: i32):
// CHECK-NEXT:     arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK-NEXT:     cf.cond_br %{{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^bb3(%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb2]](%{{.*}}: i32)
// CHECK-NEXT:     arith.addi %{{.*}}, %{{.*}}
// CHECK-NEXT:     cf.br ^[[bb3:.*]](%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb3]](%{{.*}}: i32)
// CHECK-NEXT:     cf.br ^[[bb4:.*]](%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb4]](%{{.*}}: i32)
// CHECK-NEXT:     tensor.from_elements %{{.*}} : tensor<i32>
// CHECK-NEXT:     return %{{.*}}
// CHECK-NEXT:   }

// -----

#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func @main() -> (tensor<i32>) attributes {} {
  %c0 = arith.constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<i32>
  %c10 = arith.constant 10 : i32
  %1 = tensor.from_elements %c10 : tensor<i32>
  cf.br ^bb1(%0 : tensor<i32>)

^bb1(%2: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %3 = linalg.init_tensor [] : tensor<i1>
  %4 = linalg.generic #attrs
    ins(%2, %1 : tensor<i32>, tensor<i32>)
    outs(%3 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %5 = tensor.extract %4[] : tensor<i1>
  // This cf.cond_br intentionally has bb2 as it's target for both branches. This
  // is to make sure that the "forward phase" of the cost-model correctly adds
  // the users of a block argument (in this case bb2's argument) to the work
  // list.
  cf.cond_br %5, ^bb2(%2 : tensor<i32>), ^bb2(%2 : tensor<i32>)

^bb2(%6: tensor<i32>):  // pred: ^bb1
  %12 = tensor.from_elements %c10 : tensor<i32>
  %7 = linalg.init_tensor [] : tensor<i32>
  %8 = linalg.generic #attrs
    ins(%6, %12 : tensor<i32>, tensor<i32>)
    outs(%7 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  
      %9 = arith.addi %arg0, %arg1 : i32
      linalg.yield %9 : i32
  } -> tensor<i32>
  cf.br ^bb3(%8 : tensor<i32>)

^bb3(%10: tensor<i32>):  // pred: ^bb1
  return %10 : tensor<i32>
}

// CHECK-LABEL:  func @main()
// CHECK-NEXT:     arith.constant 0
// CHECK-NEXT:     arith.constant 10
// CHECK-NEXT:     cf.br ^[[bb1:.*]](%{{.*}}: i32)
// CHECK-NEXT:   ^[[bb1]](%{{.*}}: i32):
// CHECK-NEXT:     arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK-NEXT:     cf.cond_br %{{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^bb2(%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb2]](%{{.*}}: i32)
// CHECK-NEXT:     arith.addi %{{.*}}, %{{.*}}
// CHECK-NEXT:     cf.br ^[[bb3:.*]](%{{.*}} : i32)
// CHECK-NEXT:   ^[[bb3]](%{{.*}}: i32)
// CHECK-NEXT:     tensor.from_elements %{{.*}} : tensor<i32>
// CHECK-NEXT:     return %{{.*}}
// CHECK-NEXT:   }
