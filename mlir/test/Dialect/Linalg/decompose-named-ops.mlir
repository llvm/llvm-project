// RUN: mlir-opt %s -split-input-file -linalg-decompose-named-ops | FileCheck %s
// RUN: mlir-opt %s -split-input-file -linalg-decompose-named-ops -linalg-generalize-named-ops | FileCheck %s --check-prefix=GENERALIZECHECK

func.func @softmax(%arg0: tensor<2x16x32xf32>, %dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>, %[[DST:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-DAG:  %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[INF:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-DAG:  %[[EMP:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-DAG:  %[[FIL:.+]] = linalg.fill
// CHECK-NEXT: %[[RED:.+]] = linalg.reduce ins(%[[ARG0]] : tensor<2x16x32xf32>)
// CHECK-SAME:  outs(%[[FIL]] : tensor<2x16xf32>) dimensions = [2]
// CHECK-NEXT: (%[[IN:.+]]: f32, %[[INIT:.+]]: f32) {
// CHECK-NEXT: %[[MAX:.+]] = arith.maxnumf %[[IN]], %[[INIT]] : f32
// CHECK-NEXT: linalg.yield %[[MAX]] : f32
// CHECK:      %[[CST:.+]] = linalg.broadcast ins(%[[RED]] : tensor<2x16xf32>)
// CHECK-NEXT: %[[SUB:.+]] = linalg.sub ins(%[[ARG0]], %[[CST]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// CHECK-NEXT: %[[EXP:.+]] = linalg.exp ins(%[[SUB]] : tensor<2x16x32xf32>)
// CHECK-DAG:  %[[FIL:.+]] = linalg.fill
// CHECK-NEXT: %[[SUM:.+]] = linalg.reduce ins(%[[EXP]] : tensor<2x16x32xf32>)
// CHECK-SAME:  outs(%[[FIL]] : tensor<2x16xf32>) dimensions = [2]
// CHECK-NEXT: (%[[IN:.+]]: f32, %[[INIT:.+]]: f32) {
// CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK-NEXT: linalg.yield %[[ADD]] : f32
// CHECK-DAG:  %[[EMP:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK-DAG:  %[[CST2:.+]] = linalg.broadcast ins(%[[SUM]] : tensor<2x16xf32>)
// CHECK-NEXT: %[[DIV:.+]] = linalg.div ins(%[[EXP]], %[[CST2]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) outs(%[[DST]] : tensor<2x16x32xf32>)
// CHECK: return %[[DIV]]


// GENERALIZECHECK-DAG:    #[[$MAP0:.*]] = affine_map<(d0, d1) -> ()>
// GENERALIZECHECK-DAG:    #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// GENERALIZECHECK-DAG:    #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// GENERALIZECHECK-DAG:    #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// GENERALIZECHECK-LABEL: func @softmax
// GENERALIZECHECK-SAME:     (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>, %[[DST:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// GENERALIZECHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// GENERALIZECHECK-DAG:     %[[INF:.+]] = arith.constant -3.40282347E+38 : f32
// GENERALIZECHECK-DAG:     %[[EMP:.+]] = tensor.empty() : tensor<2x16xf32>
// GENERALIZECHECK-DAG:     %[[FIL:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[INF]] : f32) outs(%[[EMP]] : tensor<2x16xf32>) {
// GENERALIZECHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      linalg.yield %[[IN]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16xf32>
// GENERALIZECHECK:         %[[RED:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]}
// GENERALIZECHECK-SAME:      ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[FIL]] : tensor<2x16xf32>) {
// GENERALIZECHECK-NEXT:    ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      %[[MAX:.+]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// GENERALIZECHECK-NEXT:      linalg.yield %[[MAX]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16xf32>
// GENERALIZECHECK:         %[[CST:.+]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP2]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[RED]] : tensor<2x16xf32>)
// GENERALIZECHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:        linalg.yield %[[IN]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16x32xf32>
// GENERALIZECHECK:         %[[SUB:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]], #[[$MAP2]]]
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[ARG0]], %[[CST]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// GENERALIZECHECK-SAME:      outs(%[[DST]] : tensor<2x16x32xf32>) {
// GENERALIZECHECK-NEXT:      ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      %[[SUBF:.+]] = arith.subf %[[LHS]], %[[RHS]] : f32
// GENERALIZECHECK-NEXT:      linalg.yield %[[SUBF]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16x32xf32>
// GENERALIZECHECK:         %[[EXP:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]]
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[SUB]] : tensor<2x16x32xf32>)
// GENERALIZECHECK-SAME:      outs(%[[DST]] : tensor<2x16x32xf32>) {
// GENERALIZECHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      %[[EXPF:.+]] = math.exp %[[IN]] : f32
// GENERALIZECHECK-NEXT:      linalg.yield %[[EXPF]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16x32xf32>
// GENERALIZECHECK:         %[[FIL:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[ZERO]] : f32) outs(%[[EMP]] : tensor<2x16xf32>) {
// GENERALIZECHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      linalg.yield %[[IN]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16xf32>
// GENERALIZECHECK:         %[[RED:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]}
// GENERALIZECHECK-SAME:      ins(%[[EXP]] : tensor<2x16x32xf32>) outs(%[[FIL]] : tensor<2x16xf32>) {
// GENERALIZECHECK-NEXT:    ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      %[[ADDF:.+]] = arith.addf %[[IN]], %[[OUT]] : f32
// GENERALIZECHECK-NEXT:      linalg.yield %[[ADDF]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16xf32>
// GENERALIZECHECK:         %[[CST:.+]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP2]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[RED]] : tensor<2x16xf32>)
// GENERALIZECHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:        linalg.yield %[[IN]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16x32xf32>
// GENERALIZECHECK:         %[[DIV:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]], #[[$MAP2]]],
// GENERALIZECHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// GENERALIZECHECK-SAME:      ins(%[[EXP]], %[[CST]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// GENERALIZECHECK-SAME:      outs(%[[DST]] : tensor<2x16x32xf32>) {
// GENERALIZECHECK-NEXT:      ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %[[OUT:.+]]: f32):
// GENERALIZECHECK-NEXT:      %[[DIVF:.+]] = arith.divf %[[LHS]], %[[RHS]] : f32
// GENERALIZECHECK-NEXT:      linalg.yield %[[DIVF]] : f32
// GENERALIZECHECK-NEXT:    } -> tensor<2x16x32xf32>
// GENERALIZECHECK:         return %[[DIV]] : tensor<2x16x32xf32>

// COM: decomposition assumes tensors as inputs, this is just to make sure nothing breaks
func.func @softmax_memref(%arg0: memref<16x64x256xf32>, %arg1: memref<16x64x256xf32>) {
  linalg.softmax
    dimension(1) ins(%arg0 : memref<16x64x256xf32>) outs(%arg1 : memref<16x64x256xf32>)
  return
}
