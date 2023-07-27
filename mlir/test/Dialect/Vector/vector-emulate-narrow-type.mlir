// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=4 memref-load-bitwidth=8" %s | FileCheck %s

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 floordiv 2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0, s1, s2, s3] -> ((s0 * s1 + s2 * s3) floordiv 2)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<()[s0, s1] -> (s0 * s1)>

// Expect no conversions, i8 is supported.
// CHECK-LABEL: func @vector_load_i8
// CHECK-SAME:  (%[[ARG:.*]]: memref<3x4xi8>, %[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
// CHECK-NEXT:  [[L:%.+]] = vector.load %[[ARG]][%[[IDX0]], %[[IDX1]]] : memref<3x4xi8>, vector<4xi8>
// CHECK-NEXT:  return
func.func @vector_load_i8(%arg0: memref<3x4xi8>, %arg1: index, %arg2: index) {
    %0 = vector.load %arg0[%arg1, %arg2] : memref<3x4xi8>, vector<4xi8>
    return
}

// -----

// CHECK-LABEL: func @vector_load_i4
// CHECK-SAME:  (%[[ARG:.*]]: memref<3x4xi8>, %[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
// CHECK-NEXT:  %[[CST:.*]] = arith.constant dense<0> : vector<3x4xi4>
// CHECK-NEXT:  %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]] : memref<3x4xi8> -> memref<i8>, index, index, index, index, index
// CHECK-NEXT:  %[[INDEX:.*]] = affine.apply #[[$MAP2]]()[%[[IDX0]], %[[STRIDES]]#0, %[[IDX1]], %[[STRIDES]]#1]
// CHECK-NEXT:  %[[LSIZE:.*]] = affine.apply #[[$MAP3]]()[%[[SIZES]]#0, %[[SIZES]]#1]
// CHECK-NEXT:  %[[AOFF:.*]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
// CHECK-NEXT:  %[[CAST:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[AOFF]]], sizes: [%[[LSIZE]]], strides: [%[[STRIDES]]#1] : memref<i8> to memref<12xi8, strided<[1], offset: ?>>
// CHECK-NEXT:  %[[LOAD:.*]] = vector.load %[[CAST]][%[[INDEX]]] : memref<12xi8, strided<[1], offset: ?>>, vector<2xi8>
// CHECK-NEXT:  %[[BITCAST:.*]] = vector.bitcast %[[LOAD]] : vector<2xi8> to vector<4xi4>
// CHECK-NEXT:  %[[INSERT:.*]] = vector.insert %[[BITCAST]], %[[CST]] [0] : vector<4xi4> into vector<3x4xi4>
// CHECK-NEXT:  return
func.func @vector_load_i4(%arg0: memref<3x4xi4>, %arg1: index, %arg2: index) {
    %cst = arith.constant dense<0> : vector<3x4xi4>
    %0 = vector.load %arg0[%arg1, %arg2] : memref<3x4xi4>, vector<4xi4>
    %1 = vector.insert %0, %cst [0] : vector<4xi4> into vector<3x4xi4>
    return
}
