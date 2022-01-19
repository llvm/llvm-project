// RUN: mlir-opt %s -pass-pipeline="builtin.func(linalg-detensorize{aggressive-mode})" | FileCheck %s -check-prefix=DET-ALL
// RUN: mlir-opt %s -pass-pipeline="builtin.func(linalg-detensorize)" | FileCheck %s -check-prefix=DET-CF

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
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %3 = tensor.extract %2[] : tensor<i1>
  cond_br %3, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)

^bb2(%4: tensor<i32>):  // pred: ^bb1
  %5 = linalg.init_tensor [] : tensor<i32>
  %6 = linalg.generic #attrs
    ins(%4, %4 : tensor<i32>, tensor<i32>)
    outs(%5 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  
      %8 = arith.addi %arg0, %arg1 : i32
      linalg.yield %8 : i32
  } -> tensor<i32>
  br ^bb1(%6 : tensor<i32>)

^bb3(%7: tensor<i32>):  // pred: ^bb1
  return %7 : tensor<i32>
}

// Test aggresively detensoring all detensorable ops.
//
// DET-ALL-LABEL: func @main
// DET-ALL-SAME:    (%{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>)
// DET-ALL:         tensor.extract {{.*}}
// DET-ALL:         br ^[[bb1:.*]](%{{.*}} : i32)
// DET-ALL:       ^[[bb1]](%{{.*}}: i32)
// DET-ALL:         arith.cmpi slt, {{.*}}
// DET-ALL:         cond_br {{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^[[bb3:.*]](%{{.*}} : i32)
// DET-ALL:       ^[[bb2]](%{{.*}}: i32)
// DET-ALL:         arith.addi {{.*}}
// DET-ALL:         br ^[[bb1]](%{{.*}} : i32)
// DET-ALL:       ^[[bb3]](%{{.*}}: i32)
// DET-ALL:         tensor.from_elements {{.*}}
// DET-ALL:         return %{{.*}} : tensor<i32>

// Test detensoring only ops involed in control-flow.
//
// DET-CF-LABEL: func @main
// DET-CF-SAME:    (%{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>)
// DET-CF:         tensor.extract {{.*}}
// DET-CF:         br ^[[bb1:.*]](%{{.*}} : i32)
// DET-CF:       ^[[bb1]](%{{.*}}: i32)
// DET-CF:         arith.cmpi slt, {{.*}}
// DET-CF:         cond_br {{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^[[bb3:.*]](%{{.*}} : i32)
// DET-CF:       ^[[bb2]](%{{.*}}: i32)
// DET-CF:         arith.addi {{.*}}
// DET-CF:         br ^[[bb1]](%{{.*}} : i32)
// DET-CF:       ^[[bb3]](%{{.*}}: i32)
// DET-CF:         tensor.from_elements %{{.*}} : tensor<i32>
// DET-CF:         return %{{.*}} : tensor<i32>
