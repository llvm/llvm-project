// RUN: mlir-opt %s -affine-super-vectorizer-test=vectorize-affine-loop-nest -split-input-file 2>&1 | FileCheck %s

module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : i32
    %sum = affine.for %i = 0 to 10 iter_args(%acc = %c0) -> i32 {
      %inc = arith.addi %acc, %c0 : i32
      affine.yield %inc : i32
    }
    return %sum : i32
  }
}

// CHECK-LABEL: func.func @main
// CHECK:         %[[vzero:.*]] = arith.constant dense<0> : vector<4xi32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 10 step 4 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<4xi32>) {
// CHECK:           %[[tail:.*]] = arith.constant dense<0> : vector<4xi32>
// CHECK:           %[[rem:.*]] = affine.apply #map(%{{.*}})
// CHECK:           %[[mask:.*]] = vector.create_mask %[[rem]] : vector<4xi1>
// CHECK:           %[[sel:.*]] = arith.select %[[mask]], %[[tail]], %[[vzero]] : vector<4xi1>, vector<4xi32>
// CHECK:           %[[add:.*]] = arith.addi %[[red_iter]], %[[sel]] : vector<4xi32>
// CHECK:           affine.yield %[[add]] : vector<4xi32>
// CHECK:         }
// CHECK:         %{{.*}} = vector.reduction <add>, %[[vred]] : vector<4xi32> into i32
// CHECK:         return %{{.*}} : i32