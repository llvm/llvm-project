// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: @promote_in0
// CHECK-SAME:  (%[[ARG0:.+]]: tensor<?x42xf32>, %{{.*}}, %{{.*}})
// CHECK:  %[[C0:.+]] = arith.constant 0
// CHECK:  %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK:  %[[ALLOC:.+]] = bufferization.alloc_tensor(%[[DIM]]) {memory_space = 1 : i64}
// CHECK:  %[[MAT:.+]] = bufferization.materialize_in_destination %[[ARG0]] in %[[ALLOC]]
// CHECK:  linalg.matmul ins(%[[MAT]], %{{.*}}
func.func @promote_in0(%arg0: tensor<?x42xf32>, %arg1: tensor<42x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1: tensor<?x42xf32>, tensor<42x?xf32>)
                       outs(%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
        %mm = transform.structured.match ops{["linalg.matmul"]} in %root
            : (!transform.any_op) -> !transform.any_op
        %op0 = transform.get_operand %mm[0]
            : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 1 %op0 : !transform.any_value
        transform.yield
    }
}

// -----

// CHECK-LABEL: @promote_out
// CHECK-SAME: (%{{.*}}: tensor<?x42xf32>, %{{.*}}: tensor<?x42xf32>, %[[ARG2:.+]]: tensor<?x?xf32>)
func.func @promote_out(%arg0: tensor<?x42xf32>, %arg1: tensor<?x42xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // CHECK:  %[[C0:.+]] = arith.constant 0
    // CHECK:  %[[DIM0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
    // CHECK:  %[[C1:.+]] = arith.constant 1
    // CHECK:  %[[DIM1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
    // CHECK:  %[[ALLOC:.+]] = bufferization.alloc_tensor(%[[DIM0]], %[[DIM1]]) {memory_space = 1 : i64}
    // CHECK-NOT: materialize_in_destination
    // CHECK:  linalg.add {{.*}} outs(%[[ALLOC]]
    %0 = linalg.add ins(%arg0, %arg1 : tensor<?x42xf32>, tensor<?x42xf32>)
                    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
        %la = transform.structured.match ops{["linalg.add"]} in %root
            : (!transform.any_op) -> !transform.any_op
        %init = transform.get_operand %la[2]
                : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 1 %init : !transform.any_value

        transform.yield
    }
}

// -----

// CHECK-LABEL: @promote_in0_out_bufferize
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x42xf32>, %{{.*}}: tensor<42x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>)
func.func @promote_in0_out_bufferize(%arg0: tensor<?x42xf32>, %arg1: tensor<42x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // CHECK:  %[[IN1:.+]] = bufferization.to_buffer %arg1 : tensor<42x?xf32> to memref<42x?xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %[[IN0:.+]] = bufferization.to_buffer %arg0 : tensor<?x42xf32> to memref<?x42xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %{{.+}} = bufferization.to_buffer %arg0 : tensor<?x42xf32> to memref<?x42xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %{{.+}} = bufferization.to_buffer %arg2 : tensor<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %{{.+}} = bufferization.to_buffer %arg2 : tensor<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %[[C0:.+]] = arith.constant 0 : index
    // CHECK:  %{{.+}} = memref.dim %{{.+}}, %[[C0]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %[[C1:.+]] = arith.constant 1 : index
    // CHECK:  %{{.+}} = memref.dim %{{.+}}, %[[C1]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %[[ALLOC_OUT:.+]] = memref.alloc(%{{.+}}, %{{.+}}) {alignment = 64 : i64} : memref<?x?xf32, 1>
    // CHECK:  %{{.+}} = arith.constant 0 : index
    // CHECK:  %{{.+}} = memref.dim %{{.+}}, %{{.+}} : memref<?x42xf32, strided<[?, ?], offset: ?>>
    // CHECK:  %[[ALLOC_IN:.+]] = memref.alloc(%{{.+}}) {alignment = 64 : i64} : memref<?x42xf32, 1>
    // CHECK:  memref.copy %[[IN0]], %[[ALLOC_IN]] : memref<?x42xf32, strided<[?, ?], offset: ?>> to memref<?x42xf32, 1>
    // CHECK: linalg.add ins(%[[ALLOC_IN]], %[[IN1]] : memref<?x42xf32, 1>, memref<42x?xf32, strided<[?, ?], offset: ?>>) outs(%[[ALLOC_OUT]] : memref<?x?xf32, 1>)
    %0 = linalg.add ins(%arg0, %arg1: tensor<?x42xf32>, tensor<42x?xf32>)
                    outs(%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
        %la = transform.structured.match ops{["linalg.add"]} in %root
            : (!transform.any_op) -> !transform.any_op
        %op0 = transform.get_operand %la[0]
            : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 1 %op0 : !transform.any_value

        %init = transform.get_operand %la[2]
                : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 1 %init : !transform.any_value

        %func = transform.structured.match ops{["func.func"]} in %root
                : (!transform.any_op) -> !transform.any_op

        %bufferized = transform.bufferization.one_shot_bufferize %func
            : (!transform.any_op) -> !transform.any_op

        transform.yield
    }
}



