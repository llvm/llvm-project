// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline="builtin.module(func.func(linalg-detensorize))" | FileCheck %s

#map = affine_map<() -> ()>
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%0 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<f32>
  cf.br ^bb1(%1 : tensor<f32>)
^bb1(%2: tensor<f32>):  // pred: ^bb0
  return %2 : tensor<f32>
}

// CHECK-LABEL: @main
// CHECK-SAME:       (%[[ARG0:.+]]: tensor<f32>) -> tensor<f32>
// CHECK:   %[[EXTRACTED:.+]] = tensor.extract %[[ARG0]][] : tensor<f32>
// CHECK: cf.br ^{{.*}}
// CHECK: ^{{.*}}:
// CHECK:   %[[ELEMENTS:.+]] = tensor.from_elements %[[EXTRACTED]] : tensor<f32>
// CHECK:   return %[[ELEMENTS]] : tensor<f32>


module {
  memref.global "private" constant @__constant_4x4xf32 : memref<4x4xf32> = dense<8.899000e+01> {alignment = 64 : i64}
  func.func private @parallel_compute_fn_with_aligned_loops(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: memref<4x4xf32>, %arg11: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb4
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb2(%0 : index), ^bb6
  ^bb2(%2: index):  // pred: ^bb1
    %3 = arith.addi %2, %c1 : index
    cf.br ^bb3(%c0 : index)
  ^bb3(%4: index):  // 2 preds: ^bb2, ^bb4
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%3 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}

// CHECK-LABEL: func.func private @parallel_compute_fn_with_aligned_loops(
// CHECK-SAME:    %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, {{.*}}) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           cf.br ^bb1(%[[C0]] : index)
// CHECK:         ^bb1(%[[VAL_0:.*]]: index):
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[C4]] : index
// CHECK:           cf.cond_br %[[CMPI_0]], ^bb2(%[[VAL_0]] : index), ^bb5
// CHECK:         ^bb2(%[[VAL_1:.*]]: index):
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[C1]] : index
// CHECK:           cf.br ^bb3(%[[C0]] : index)
// CHECK:         ^bb3(%[[VAL_2:.*]]: index):
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_2]], %[[C4]] : index
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           cf.br ^bb1(%[[ADDI_0]] : index)
// CHECK:         ^bb5:
// CHECK:           return