// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

func @generalize_matmul_tensor_f32(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----

func @generalize_matmul_tensor_i32(%A : tensor<16x8xi32>, %B: tensor<8x32xi32>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xi32>, tensor<8x32xi32>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i32, %[[B_ARG:.+]]: i32, %[[C_ARG:.+]]: i32)
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_ARG]], %[[B_ARG]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>
