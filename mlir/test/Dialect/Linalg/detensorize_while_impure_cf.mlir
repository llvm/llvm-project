// RUN: mlir-opt %s -pass-pipeline="builtin.func(linalg-detensorize{aggressive-mode})" | FileCheck %s -check-prefix=DET-ALL
// RUN: mlir-opt %s -pass-pipeline="builtin.func(linalg-detensorize)" | FileCheck %s -check-prefix=DET-CF

#map0 = affine_map<() -> ()>
#map1 = affine_map<(i) -> ()>
#map2 = affine_map<(i) -> (i)>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

#sum_reduction_attrs = {
  indexing_maps = [#map2, #map1],
  iterator_types = ["reduction"]
}


#broadcast_attrs = {
  indexing_maps = [#map1, #map2],
  iterator_types = ["parallel"]
}

func @main(%farg0: tensor<10xi32>, %farg1: tensor<i32>) -> tensor<i32> attributes {} {
  br ^bb1(%farg0 : tensor<10xi32>)

^bb1(%0: tensor<10xi32>):  // 2 preds: ^bb0, ^bb2
  %1 = linalg.init_tensor [] : tensor<i32>
  %2 = linalg.generic #sum_reduction_attrs
    ins(%0: tensor<10xi32>)
    outs(%1: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        %b = arith.addi %x, %a : i32
        linalg.yield %b : i32
  } -> tensor<i32>

  %3 = linalg.init_tensor [] : tensor<i1>
  %4 = linalg.generic #attrs
    ins(%2, %farg1 : tensor<i32>, tensor<i32>)
    outs(%3 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):  
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %5 = tensor.extract %4[] : tensor<i1>
  cond_br %5, ^bb2(%2 : tensor<i32>), ^bb3(%2 : tensor<i32>)

^bb2(%6: tensor<i32>):  // pred: ^bb1
  %7 = linalg.init_tensor [10] : tensor<10xi32>
  %9 = linalg.generic #broadcast_attrs
       ins(%6: tensor<i32>)
      outs(%7: tensor<10xi32>) {
    ^bb(%a: i32, %b: i32) :
      linalg.yield %a : i32
  } -> tensor<10xi32>

  br ^bb1(%9 : tensor<10xi32>)

^bb3(%10: tensor<i32>):  // pred: ^bb1
  return %10 : tensor<i32>
}

// Test aggresively detensoring all detensorable ops.
//
// DET-ALL-LABEL: func @main
// DET-ALL-SAME:    (%{{.*}}: tensor<10xi32>, %{{.*}}: tensor<i32>)
// DET-ALL:         br ^[[bb1:.*]](%{{.*}} : tensor<10xi32>)
// DET-ALL:       ^[[bb1]](%{{.*}}: tensor<10xi32>)
// DET-ALL:         linalg.init_tensor [] : tensor<i32>
// DET-ALL:         linalg.generic {{{.*}}} ins(%{{.*}} : tensor<10xi32>) outs(%{{.*}} : tensor<i32>) {
// DET-ALL:         ^bb0(%{{.*}}: i32, %{{.*}}: i32):  
// DET-ALL:           %{{.*}} = arith.addi %{{.*}}, %{{.*}}
// DET-ALL:           linalg.yield %{{.*}} : i32
// DET-ALL:         } -> tensor<i32>
// DET-ALL:         tensor.extract %{{.*}}[] : tensor<i32>
// DET-ALL:         cmpi slt, %{{.*}}, %{{.*}} : i32
// DET-ALL:         cond_br %{{.*}}, ^[[bb2:.*]](%{{.*}} : i32), ^[[bb3:.*]](%{{.*}} : i32)
// DET-ALL:       ^[[bb2]](%{{.*}}: i32)
// DET-ALL:         tensor.from_elements %{{.*}} : tensor<i32>
// DET-ALL:         linalg.init_tensor [10] : tensor<10xi32>
// DET-ALL:         linalg.generic {{{.*}}} ins(%{{.*}} : tensor<i32>) outs(%{{.*}} : tensor<10xi32>) {
// DET-ALL:         ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// DET-ALL:           linalg.yield %{{.*}} : i32
// DET-ALL:         } -> tensor<10xi32>
// DET-ALL:         br ^[[bb1]](%{{.*}} : tensor<10xi32>)
// DET-ALL:       ^[[bb3]](%{{.*}}: i32)
// DET-ALL:         tensor.from_elements %{{.*}} : tensor<i32>
// DET-ALL:         return %{{.*}} : tensor<i32>
// DET-ALL:       }

// DET-CF-LABEL: func @main
// DET-CF-SAME:    (%{{.*}}: tensor<10xi32>, %{{.*}}: tensor<i32>)
// DET-CF:         br ^[[bb1:.*]](%{{.*}} : tensor<10xi32>)
// DET-CF:       ^bb1(%{{.*}}: tensor<10xi32>)
// DET-CF:         %{{.*}} = linalg.generic {{{.*}}} ins(%{{.*}} : tensor<10xi32>) outs(%{{.*}} : tensor<i32>) {
// DET-CF:         tensor.extract %{{.*}}[] : tensor<i32>
// DET-CF:         cmpi slt, %{{.*}}, %{{.*}} : i32
// DET-CF:         cond_br %{{.*}}, ^bb2(%{{.*}} : tensor<i32>), ^bb3(%{{.*}} : tensor<i32>)
// DET-CF:       ^bb2(%{{.*}}: tensor<i32>)
// DET-CF:         %{{.*}} = linalg.generic {{{.*}}} ins(%{{.*}} : tensor<i32>) outs(%{{.*}} : tensor<10xi32>) {
// DET-CF:         br ^bb1(%{{.*}} : tensor<10xi32>)
// DET-CF:       ^bb3(%{{.*}}: tensor<i32>)
// DET-CF:         return %{{.*}} : tensor<i32>
// DET-CF:       }
