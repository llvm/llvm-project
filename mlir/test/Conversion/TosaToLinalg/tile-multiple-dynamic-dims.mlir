// RUN: mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg))' %s | FileCheck %s

// CHECK-LABEL: func.func @tile_dynamic(
// CHECK: linalg.generic
// CHECK: tensor.collapse_shape {{.*}} : tensor<2x?x3x?xf32> into tensor<?x?xf32>
// CHECK-NOT: tosa.reshape
func.func @tile_dynamic(%a: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %multiples = tosa.const_shape values(dense<[2, 3]> : tensor<2xindex>)
      : () -> !tosa.shape<2>
  %tile = tosa.tile %a, %multiples : (tensor<?x?xf32>, !tosa.shape<2>) -> tensor<?x?xf32>
  return %tile : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @tile_partly_static(
// CHECK: %[[SHAPE:.*]] = tensor.collapse_shape {{.*}} : tensor<2x2x3x?xf32> into tensor<4x?xf32>
// CHECK: tensor.cast %[[SHAPE]] : tensor<4x?xf32> to tensor<?x?xf32>
func.func @tile_partly_static(%a: tensor<2x?xf32>) -> tensor<?x?xf32> {
  %multiples = tosa.const_shape values(dense<[2, 3]> : tensor<2xindex>)
      : () -> !tosa.shape<2>
  %tile = tosa.tile %a, %multiples : (tensor<2x?xf32>, !tosa.shape<2>) -> tensor<?x?xf32>
  return %tile : tensor<?x?xf32>
}
