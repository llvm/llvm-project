// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --cse --split-input-file %s | FileCheck %s
// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=32" --cse --split-input-file %s | FileCheck %s --check-prefix=CHECK32

func.func @vector_load_i8(%arg1: index, %arg2: index) -> vector<4xi8> {
    %0 = memref.alloc() : memref<3x4xi8>
    %1 = vector.load %0[%arg1, %arg2] : memref<3x4xi8>, vector<4xi8>
    return %1 : vector<4xi8>
}
// Expect no conversions, i8 is supported.
//      CHECK: func @vector_load_i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index)
// CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<3x4xi8>
// CHECK-NEXT:   [[L:%.+]] = vector.load %[[ALLOC]][%[[ARG0]], %[[ARG1]]] : memref<3x4xi8>, vector<4xi8>
// CHECK-NEXT:   return

//      CHECK32: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 4)>
//      CHECK32: func @vector_load_i8(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index)
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[VECLOAD:.+]] = vector.load %[[ALLOC]][%[[INDEX]]] : memref<3xi32>, vector<1xi32>
//      CHECK32:   %[[VEC_I4:.+]] = vector.bitcast %[[VECLOAD]] : vector<1xi32> to vector<4xi8>
//      CHECK32:   return %[[VEC_I4]]

// -----

func.func @vector_load_i4(%arg1: index, %arg2: index) -> vector<3x8xi4> {
    %0 = memref.alloc() : memref<3x8xi4>
    %cst = arith.constant dense<0> : vector<3x8xi4>
    %1 = vector.load %0[%arg1, %arg2] : memref<3x8xi4>, vector<8xi4>
    %2 = vector.insert %1, %cst [0] : vector<8xi4> into vector<3x8xi4>
    return %2 : vector<3x8xi4>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 4 + s1 floordiv 2)>
//      CHECK: func @vector_load_i4
// CHECK-SAME:     (%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index)
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<12xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[VEC:.+]] = vector.load %[[ALLOC]][%[[INDEX]]] : memref<12xi8>, vector<4xi8>
//      CHECK:   %[[VEC_I4:.+]] = vector.bitcast %[[VEC]] : vector<4xi8> to vector<8xi4>

//  CHECK32-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 8)>
//      CHECK32: func @vector_load_i4
// CHECK32-SAME:     (%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index)
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[VEC:.+]] = vector.load %[[ALLOC]][%[[INDEX]]] : memref<3xi32>, vector<1xi32>
//      CHECK32:   %[[VEC_I4:.+]] = vector.bitcast %[[VEC]] : vector<1xi32> to vector<8xi4>

// -----

func.func @vector_load_i4_dynamic(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) -> vector<8xi4> {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xi4>
  %1 = vector.load %0[%arg2, %arg3] : memref<?x?xi4>, vector<8xi4>
  return %1 : vector<8xi4>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 2)>
//      CHECK: func.func @vector_load_i4_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//      CHECK:   %[[VEC:.+]] = vector.load %[[ALLOC]][%[[INDEX]]] : memref<?xi8>, vector<4xi8>
//      CHECK:   %[[VEC_I4:.+]] = vector.bitcast %[[VEC]] : vector<4xi8> to vector<8xi4>

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 8)>
//      CHECK32: func.func @vector_load_i4_dynamic(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK32-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK32-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK32:   %[[SIZE:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi32>
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//      CHECK32:   %[[VEC:.+]] = vector.load %[[ALLOC]][%[[INDEX]]] : memref<?xi32>, vector<1xi32>
//      CHECK32:   %[[VEC_I4:.+]] = vector.bitcast %[[VEC]] : vector<1xi32> to vector<8xi4>

// -----

func.func @vector_transfer_read_i4(%arg1: index, %arg2: index) -> vector<8xi4> {
    %c0 = arith.constant 0 : i4
    %0 = memref.alloc() : memref<3x8xi4>
    %1 = vector.transfer_read %0[%arg1, %arg2], %c0 {in_bounds = [true]} :
      memref<3x8xi4>, vector<8xi4>
    return %1 : vector<8xi4>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 4 + s1 floordiv 2)>
//      CHECK: func @vector_transfer_read_i4
// CHECK-SAME:     (%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index)
//      CHECK:   %[[CONST:.+]] = arith.constant 0 : i4
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<12xi8>
//      CHECK:   %[[PAD:.+]] = arith.extui %[[CONST]] : i4 to i8
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[ALLOC]][%[[INDEX]]], %[[PAD]] : memref<12xi8>, vector<4xi8>
//      CHECK:   %[[VEC_I4:.+]] = vector.bitcast %[[VEC]] : vector<4xi8> to vector<8xi4>

//  CHECK32-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 8)>
//      CHECK32: func @vector_transfer_read_i4
// CHECK32-SAME:     (%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index)
//      CHECK32:   %[[CONST:.+]] = arith.constant 0 : i4
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[PAD:.+]] = arith.extui %[[CONST]] : i4 to i32
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[VEC:.+]] = vector.transfer_read %[[ALLOC]][%[[INDEX]]], %[[PAD]] : memref<3xi32>, vector<1xi32>
//      CHECK32:   %[[VEC_I4:.+]] = vector.bitcast %[[VEC]] : vector<1xi32> to vector<8xi4>
