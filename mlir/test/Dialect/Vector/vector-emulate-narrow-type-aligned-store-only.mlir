// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 assume-aligned=true" --cse --verify-diagnostics --split-input-file %s | FileCheck %s

/// Aligned store, constant index - the source vector fills whole container
/// elements. Produces a simple bitcast + store.
func.func @vector_store_i4_aligned_const(%arg0: vector<8xi4>, %arg1: index, %arg2: index) {
    %0 = memref.alloc() : memref<4x8xi4>
    vector.store %arg0, %0[%arg1, %arg2] : memref<4x8xi4>, vector<8xi4>
    return
}
//  CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 4 + s1 floordiv 2)>
//      CHECK: func @vector_store_i4_aligned_const
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: vector<8xi4>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<16xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK:   %[[VEC_I8:.+]] = vector.bitcast %[[ARG0]] : vector<8xi4> to vector<4xi8>
//      CHECK:   vector.store %[[VEC_I8]], %[[ALLOC]][%[[INDEX]]] : memref<16xi8>, vector<4xi8>

// -----

/// Aligned store, dynamic index. The source vector (8 x i4 = 32 bits) is a
/// whole multiple of the container element size (i8 = 8 bits), so no partial
/// stores are needed. This holds regardless of the dynamic offset.
func.func @vector_store_i4_aligned_dynamic(%arg0: vector<8xi4>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    %0 = memref.alloc(%arg1, %arg2) : memref<?x?xi4>
    vector.store %arg0, %0[%arg3, %arg4] : memref<?x?xi4>, vector<8xi4>
    return
}
//  CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2, s0 floordiv 2)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 2)>
//      CHECK: func @vector_store_i4_aligned_dynamic
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: vector<8xi4>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[SIZE:.+]] = affine.max #[[$MAP]]()[%[[ARG2]], %[[ARG1]]]
//      CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[ARG3]], %[[ARG2]], %[[ARG4]]]
//      CHECK:   %[[VEC_I8:.+]] = vector.bitcast %[[ARG0]] : vector<8xi4> to vector<4xi8>
//      CHECK:   vector.store %[[VEC_I8]], %[[ALLOC]][%[[INDEX]]] : memref<?xi8>, vector<4xi8>

// -----

/// The source vector does not fill whole container elements (3 x i4 != N x i8),
/// so the aligned pattern rejects it. With aligned-store-only, no unaligned
/// pattern is available, so legalization fails.
func.func @vector_store_i4_not_divisible(%arg0: vector<3xi4>) {
    %0 = memref.alloc() : memref<12xi4>
    %c0 = arith.constant 0 : index
    // expected-error @below {{failed to legalize operation 'vector.store' that was explicitly marked illegal}}
    vector.store %arg0, %0[%c0] : memref<12xi4>, vector<3xi4>
    return
}
