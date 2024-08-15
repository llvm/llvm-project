// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8" --cse --split-input-file %s | FileCheck %s
// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=32" --cse --split-input-file %s | FileCheck %s --check-prefix=CHECK32

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

// -----

func.func @vector_maskedload_i8(%arg1: index, %arg2: index, %arg3: index, %passthru: vector<4xi8>) -> vector<4xi8> {
    %0 = memref.alloc() : memref<3x4xi8>
    %mask = vector.create_mask %arg3 : vector<4xi1>
    %1 = vector.maskedload %0[%arg1, %arg2], %mask, %passthru :
      memref<3x4xi8>, vector<4xi1>, vector<4xi8> into vector<4xi8>
    return %1 : vector<4xi8>
}
// Expect no conversions, i8 is supported.
//      CHECK: func @vector_maskedload_i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index, %[[ARG3:[a-zA-Z0-9]+]]: vector<4xi8>)
// CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<3x4xi8>
// CHECK-NEXT:   %[[MASK:.+]] = vector.create_mask %[[ARG2]] : vector<4xi1>
// CHECK-NEXT:   [[L:%.+]] = vector.maskedload %[[ALLOC]][%[[ARG0]], %[[ARG1]]], %[[MASK]], %[[ARG3]] :
// CHECK-SAME:     memref<3x4xi8>, vector<4xi1>, vector<4xi8> into vector<4xi8>
// CHECK-NEXT:   return

//  CHECK32-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 4)>
//  CHECK32-DAG: #[[MASK_IDX_MAP:.+]] = affine_map<()[s0] -> ((s0 + 3) floordiv 4)>
//      CHECK32: func @vector_maskedload_i8(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index, %[[ARG3:[a-zA-Z0-9]+]]: vector<4xi8>)
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[ORIG_MASK:.+]] = vector.create_mask %[[ARG2]] : vector<4xi1>
//      CHECK32:   %[[LD_IDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[MASK_IDX:.+]] = affine.apply #[[MASK_IDX_MAP]]()[%[[ARG2]]]
//      CHECK32:   %[[NEW_MASK:.+]] = vector.create_mask %[[MASK_IDX]] : vector<1xi1>
//      CHECK32:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[ARG3]] : vector<4xi8> to vector<1xi32>
//      CHECK32:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LD_IDX]]], %[[NEW_MASK]], %[[NEW_PASSTHRU]] :
// CHECK32-SAME:     memref<3xi32>, vector<1xi1>, vector<1xi32> into vector<1xi32>
//      CHECK32:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi32> to vector<4xi8>
//      CHECK32:   %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[BITCAST]], %[[ARG3]] : vector<4xi1>, vector<4xi8>
//      CHECK32:   return %[[SELECT]]

// -----

func.func @vector_maskedload_i4(%arg1: index, %arg2: index, %arg3: index, %passthru: vector<8xi4>) -> vector<3x8xi4> {
    %0 = memref.alloc() : memref<3x8xi4>
    %cst = arith.constant dense<0> : vector<3x8xi4>
    %mask = vector.create_mask %arg3 : vector<8xi1>
    %1 = vector.maskedload %0[%arg1, %arg2], %mask, %passthru :
      memref<3x8xi4>, vector<8xi1>, vector<8xi4> into vector<8xi4>
    %2 = vector.insert %1, %cst [0] : vector<8xi4> into vector<3x8xi4>
    return %2 : vector<3x8xi4>
}
//  CHECK-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 4 + s1 floordiv 2)>
//  CHECK-DAG: #[[MASK_IDX_MAP:.+]] = affine_map<()[s0] -> ((s0 + 1) floordiv 2)>
//      CHECK: func @vector_maskedload_i4(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index, %[[ARG3:[a-zA-Z0-9]+]]: vector<8xi4>)
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<12xi8>
//      CHECK:   %[[ORIG_MASK:.+]] = vector.create_mask %[[ARG2]] : vector<8xi1>
//      CHECK:   %[[LD_IDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[MASK_IDX:.+]] = affine.apply #[[MASK_IDX_MAP]]()[%[[ARG2]]]
//      CHECK:   %[[NEW_MASK:.+]] = vector.create_mask %[[MASK_IDX]] : vector<4xi1>
//      CHECK:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[ARG3]] : vector<8xi4> to vector<4xi8>
//      CHECK:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LD_IDX]]], %[[NEW_MASK]], %[[NEW_PASSTHRU]] :
// CHECK-SAME:     memref<12xi8>, vector<4xi1>, vector<4xi8> into vector<4xi8>
//      CHECK:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<4xi8> to vector<8xi4>
//      CHECK:   %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[BITCAST]], %[[ARG3]] : vector<8xi1>, vector<8xi4>

//  CHECK32-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 8)>
//  CHECK32-DAG: #[[MASK_IDX_MAP:.+]] = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
//      CHECK32: func @vector_maskedload_i4(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index, %[[ARG3:[a-zA-Z0-9]+]]: vector<8xi4>)
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[ORIG_MASK:.+]] = vector.create_mask %[[ARG2]] : vector<8xi1>
//      CHECK32:   %[[LD_IDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[MASK_IDX:.+]] = affine.apply #[[MASK_IDX_MAP]]()[%[[ARG2]]]
//      CHECK32:   %[[NEW_MASK:.+]] = vector.create_mask %[[MASK_IDX]] : vector<1xi1>
//      CHECK32:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[ARG3]] : vector<8xi4> to vector<1xi32>
//      CHECK32:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LD_IDX]]], %[[NEW_MASK]], %[[NEW_PASSTHRU]] :
// CHECK32-SAME:     memref<3xi32>, vector<1xi1>, vector<1xi32> into vector<1xi32>
//      CHECK32:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi32> to vector<8xi4>
//      CHECK32:   %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[BITCAST]], %[[ARG3]] : vector<8xi1>, vector<8xi4>

// -----

func.func @vector_cst_maskedload_i8(%arg1: index, %arg2: index, %passthru: vector<4xi8>) -> vector<4xi8> {
    %0 = memref.alloc() : memref<3x4xi8>
    %mask = vector.constant_mask [2] : vector<4xi1>
    %1 = vector.maskedload %0[%arg1, %arg2], %mask, %passthru :
      memref<3x4xi8>, vector<4xi1>, vector<4xi8> into vector<4xi8>
    return %1 : vector<4xi8>
}
// Expect no conversions, i8 is supported.
//      CHECK: func @vector_cst_maskedload_i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: vector<4xi8>)
// CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<3x4xi8>
// CHECK-NEXT:   %[[MASK:.+]] = vector.constant_mask [2] : vector<4xi1>
// CHECK-NEXT:   [[L:%.+]] = vector.maskedload %[[ALLOC]][%[[ARG0]], %[[ARG1]]], %[[MASK]], %[[ARG2]] :
// CHECK-SAME:     memref<3x4xi8>, vector<4xi1>, vector<4xi8> into vector<4xi8>
// CHECK-NEXT:   return

//  CHECK32-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 4)>
//      CHECK32: func @vector_cst_maskedload_i8(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK32-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: vector<4xi8>)
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[ORIG_MASK:.+]] = vector.constant_mask [2] : vector<4xi1>
//      CHECK32:   %[[LD_IDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[NEW_MASK:.+]] = vector.constant_mask [1] : vector<1xi1>
//      CHECK32:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[ARG3]] : vector<4xi8> to vector<1xi32>
//      CHECK32:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LD_IDX]]], %[[NEW_MASK]], %[[NEW_PASSTHRU]] :
// CHECK32-SAME:     memref<3xi32>, vector<1xi1>, vector<1xi32> into vector<1xi32>
//      CHECK32:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi32> to vector<4xi8>
//      CHECK32:   %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[BITCAST]], %[[ARG3]] : vector<4xi1>, vector<4xi8>
//      CHECK32:   return %[[SELECT]]

// -----

func.func @vector_cst_maskedload_i4(%arg1: index, %arg2: index, %passthru: vector<8xi4>) -> vector<3x8xi4> {
    %0 = memref.alloc() : memref<3x8xi4>
    %cst = arith.constant dense<0> : vector<3x8xi4>
    %mask = vector.constant_mask [4] : vector<8xi1>
    %1 = vector.maskedload %0[%arg1, %arg2], %mask, %passthru :
      memref<3x8xi4>, vector<8xi1>, vector<8xi4> into vector<8xi4>
    %2 = vector.insert %1, %cst [0] : vector<8xi4> into vector<3x8xi4>
    return %2 : vector<3x8xi4>
}
//  CHECK-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 4 + s1 floordiv 2)>
//      CHECK: func @vector_cst_maskedload_i4(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: vector<8xi4>)
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<12xi8>
//      CHECK:   %[[ORIG_MASK:.+]] = vector.constant_mask [4] : vector<8xi1>
//      CHECK:   %[[LD_IDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[NEW_MASK:.+]] = vector.constant_mask [2] : vector<4xi1>
//      CHECK:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[ARG2]] : vector<8xi4> to vector<4xi8>
//      CHECK:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LD_IDX]]], %[[NEW_MASK]], %[[NEW_PASSTHRU]] :
// CHECK-SAME:     memref<12xi8>, vector<4xi1>, vector<4xi8> into vector<4xi8>
//      CHECK:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<4xi8> to vector<8xi4>
//      CHECK:   %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[BITCAST]], %[[ARG2]] : vector<8xi1>, vector<8xi4>

//  CHECK32-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 8)>
//      CHECK32: func @vector_cst_maskedload_i4(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: vector<8xi4>)
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi32>
//      CHECK32:   %[[ORIG_MASK:.+]] = vector.constant_mask [4] : vector<8xi1>
//      CHECK32:   %[[LD_IDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[NEW_MASK:.+]] = vector.constant_mask [1] : vector<1xi1>
//      CHECK32:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[ARG2]] : vector<8xi4> to vector<1xi32>
//      CHECK32:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LD_IDX]]], %[[NEW_MASK]], %[[NEW_PASSTHRU]] :
// CHECK32-SAME:     memref<3xi32>, vector<1xi1>, vector<1xi32> into vector<1xi32>
//      CHECK32:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi32> to vector<8xi4>
//      CHECK32:   %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[BITCAST]], %[[ARG2]] : vector<8xi1>, vector<8xi4>

// -----

func.func @vector_extract_maskedload_i4(%arg1: index) -> vector<8x8x16xi4> {
    %0 = memref.alloc() : memref<8x8x16xi4>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %cst_1 = arith.constant dense<0> : vector<8x8x16xi4>
    %cst_2 = arith.constant dense<0> : vector<16xi4>
    %27 = vector.create_mask %c8, %arg1, %c16 : vector<8x8x16xi1>
    %48 = vector.extract %27[0] : vector<8x16xi1> from vector<8x8x16xi1>
    %49 = vector.extract %48[0] : vector<16xi1> from vector<8x16xi1>
    %50 = vector.maskedload %0[%c0, %c0, %c0], %49, %cst_2 : memref<8x8x16xi4>, vector<16xi1>, vector<16xi4> into vector<16xi4>
    %63 = vector.insert %50, %cst_1 [0, 0] : vector<16xi4> into vector<8x8x16xi4>
    return %63 : vector<8x8x16xi4>
}
//      CHECK: func @vector_extract_maskedload_i4(
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<512xi8>
//      CHECK:   %[[PASSTHRU:.+]] = arith.constant dense<0> : vector<16xi4>
//      CHECK:   %[[ORIG_MASK:.+]] = vector.create_mask {{.*}} vector<8x8x16xi1>
//      CHECK:   %[[ORIG_EXT1:.+]] = vector.extract %[[ORIG_MASK]][0] : vector<8x16xi1>
//      CHECK:   %[[ORIG_EXT2:.+]] = vector.extract %[[ORIG_EXT1]][0] : vector<16xi1>
//      CHECK:   %[[NEW_MASK:.+]] = vector.create_mask {{.*}} vector<8x8x8xi1>
//      CHECK:   %[[NEW_EXT1:.+]] = vector.extract %[[NEW_MASK]][0] : vector<8x8xi1>
//      CHECK:   %[[NEW_EXT2:.+]] = vector.extract %[[NEW_EXT1]][0] : vector<8xi1>
//      CHECK:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[PASSTHRU]] : vector<16xi4> to vector<8xi8>
//      CHECK:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%c0], %[[NEW_EXT2]], %[[NEW_PASSTHRU]] :
// CHECK-SAME:     memref<512xi8>, vector<8xi1>, vector<8xi8> into vector<8xi8>
//      CHECK:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<8xi8> to vector<16xi4>
//      CHECK:   %[[SELECT:.+]] = arith.select %[[ORIG_EXT2]], %[[BITCAST]], %[[PASSTHRU]] : vector<16xi1>, vector<16xi4>

//      CHECK32: func @vector_extract_maskedload_i4(
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<128xi32>
//      CHECK32:   %[[PASSTHRU:.+]] = arith.constant dense<0> : vector<16xi4>
//      CHECK32:   %[[ORIG_MASK:.+]] = vector.create_mask {{.*}} vector<8x8x16xi1>
//      CHECK32:   %[[ORIG_EXT1:.+]] = vector.extract %[[ORIG_MASK]][0] : vector<8x16xi1>
//      CHECK32:   %[[ORIG_EXT2:.+]] = vector.extract %[[ORIG_EXT1]][0] : vector<16xi1>
//      CHECK32:   %[[NEW_MASK:.+]] = vector.create_mask {{.*}} vector<8x8x2xi1>
//      CHECK32:   %[[NEW_EXT1:.+]] = vector.extract %[[NEW_MASK]][0] : vector<8x2xi1>
//      CHECK32:   %[[NEW_EXT2:.+]] = vector.extract %[[NEW_EXT1]][0] : vector<2xi1>
//      CHECK32:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[PASSTHRU]] : vector<16xi4> to vector<2xi32>
//      CHECK32:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%c0], %[[NEW_EXT2]], %[[NEW_PASSTHRU]] :
// CHECK32-SAME:     memref<128xi32>, vector<2xi1>, vector<2xi32> into vector<2xi32>
//      CHECK32:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<2xi32> to vector<16xi4>
//      CHECK32:   %[[SELECT:.+]] = arith.select %[[ORIG_EXT2]], %[[BITCAST]], %[[PASSTHRU]] : vector<16xi1>, vector<16xi4>

// -----

func.func @vector_extract_cst_maskedload_i4() -> vector<8x8x16xi4> {
    %0 = memref.alloc() : memref<8x8x16xi4>
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant dense<0> : vector<8x8x16xi4>
    %cst_2 = arith.constant dense<0> : vector<16xi4>
    %27 = vector.constant_mask [8, 4, 16] : vector<8x8x16xi1>
    %48 = vector.extract %27[0] : vector<8x16xi1> from vector<8x8x16xi1>
    %49 = vector.extract %48[0] : vector<16xi1> from vector<8x16xi1>
    %50 = vector.maskedload %0[%c0, %c0, %c0], %49, %cst_2 : memref<8x8x16xi4>, vector<16xi1>, vector<16xi4> into vector<16xi4>
    %63 = vector.insert %50, %cst_1 [0, 0] : vector<16xi4> into vector<8x8x16xi4>
    return %63 : vector<8x8x16xi4>
}
//      CHECK: func @vector_extract_cst_maskedload_i4(
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<512xi8>
//      CHECK:   %[[PASSTHRU:.+]] = arith.constant dense<0> : vector<16xi4>
//      CHECK:   %[[ORIG_MASK:.+]] = vector.constant_mask {{.*}} vector<8x8x16xi1>
//      CHECK:   %[[ORIG_EXT1:.+]] = vector.extract %[[ORIG_MASK]][0] : vector<8x16xi1>
//      CHECK:   %[[ORIG_EXT2:.+]] = vector.extract %[[ORIG_EXT1]][0] : vector<16xi1>
//      CHECK:   %[[NEW_MASK:.+]] = vector.constant_mask {{.*}} vector<8x8x8xi1>
//      CHECK:   %[[NEW_EXT1:.+]] = vector.extract %[[NEW_MASK]][0] : vector<8x8xi1>
//      CHECK:   %[[NEW_EXT2:.+]] = vector.extract %[[NEW_EXT1]][0] : vector<8xi1>
//      CHECK:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[PASSTHRU]] : vector<16xi4> to vector<8xi8>
//      CHECK:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%c0], %[[NEW_EXT2]], %[[NEW_PASSTHRU]] :
// CHECK-SAME:     memref<512xi8>, vector<8xi1>, vector<8xi8> into vector<8xi8>
//      CHECK:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<8xi8> to vector<16xi4>
//      CHECK:   %[[SELECT:.+]] = arith.select %[[ORIG_EXT2]], %[[BITCAST]], %[[PASSTHRU]] : vector<16xi1>, vector<16xi4>

//      CHECK32: func @vector_extract_cst_maskedload_i4(
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<128xi32>
//      CHECK32:   %[[PASSTHRU:.+]] = arith.constant dense<0> : vector<16xi4>
//      CHECK32:   %[[ORIG_MASK:.+]] = vector.constant_mask {{.*}} vector<8x8x16xi1>
//      CHECK32:   %[[ORIG_EXT1:.+]] = vector.extract %[[ORIG_MASK]][0] : vector<8x16xi1>
//      CHECK32:   %[[ORIG_EXT2:.+]] = vector.extract %[[ORIG_EXT1]][0] : vector<16xi1>
//      CHECK32:   %[[NEW_MASK:.+]] = vector.constant_mask {{.*}} vector<8x8x2xi1>
//      CHECK32:   %[[NEW_EXT1:.+]] = vector.extract %[[NEW_MASK]][0] : vector<8x2xi1>
//      CHECK32:   %[[NEW_EXT2:.+]] = vector.extract %[[NEW_EXT1]][0] : vector<2xi1>
//      CHECK32:   %[[NEW_PASSTHRU:.+]] = vector.bitcast %[[PASSTHRU]] : vector<16xi4> to vector<2xi32>
//      CHECK32:   %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%c0], %[[NEW_EXT2]], %[[NEW_PASSTHRU]] :
// CHECK32-SAME:     memref<128xi32>, vector<2xi1>, vector<2xi32> into vector<2xi32>
//      CHECK32:   %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<2xi32> to vector<16xi4>
//      CHECK32:   %[[SELECT:.+]] = arith.select %[[ORIG_EXT2]], %[[BITCAST]], %[[PASSTHRU]] : vector<16xi1>, vector<16xi4>

// -----

func.func @vector_store_i8(%arg0: vector<8xi8>, %arg1: index, %arg2: index) {
    %0 = memref.alloc() : memref<4x8xi8>
    vector.store %arg0, %0[%arg1, %arg2] :memref<4x8xi8>, vector<8xi8>
    return
}

// Expect no conversions, i8 is supported.
//      CHECK: func @vector_store_i8
//      CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<4x8xi8>
//      CHECK: vector.store %[[ARG0]], %[[ALLOC:.+]][%[[ARG1]], %[[ARG2]]] : memref<4x8xi8>, vector<8xi8>

//  CHECK32-DAG: affine_map<()[s0, s1] -> (s0 * 2 + s1 floordiv 4)>
//      CHECK32: func @vector_store_i8
//      CHECK32: %[[ALLOC:.+]] = memref.alloc() : memref<8xi32>
//      CHECK32: %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK32: %[[VEC_I32:.+]] = vector.bitcast %[[ARG0]] : vector<8xi8> to vector<2xi32>
//      CHECK32: vector.store %[[VEC_I32:.+]], %[[ALLOC:.+]][%[[INDEX:.+]]] : memref<8xi32>, vector<2xi32

// -----

func.func @vector_store_i4(%arg0: vector<8xi4>, %arg1: index, %arg2: index) {
    %0 = memref.alloc() : memref<4x8xi4>
    vector.store %arg0, %0[%arg1, %arg2] :memref<4x8xi4>, vector<8xi4>
    return
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 4 + s1 floordiv 2)>
//      CHECK: func @vector_store_i4
//      CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<16xi8>
//      CHECK: %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK: %[[VEC_I8:.+]] = vector.bitcast %[[ARG0]] : vector<8xi4> to vector<4xi8>
//      CHECK: vector.store %[[VEC_I8:.+]], %[[ALLOC:.+]][%[[INDEX:.+]]] : memref<16xi8>, vector<4xi8>

//  CHECK32-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 8)>
//      CHECK32: func @vector_store_i4
//      CHECK32: %[[ALLOC:.+]] = memref.alloc() : memref<4xi32>
//      CHECK32: %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK32: %[[VEC_I32:.+]] = vector.bitcast %[[ARG0]] : vector<8xi4> to vector<1xi32>
//      CHECK32: vector.store %[[VEC_I32:.+]], %[[ALLOC:.+]][%[[INDEX:.+]]] : memref<4xi32>, vector<1xi32>

// -----

func.func @vector_store_i4_dynamic(%arg0: vector<8xi4>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    %0 = memref.alloc(%arg1, %arg2) : memref<?x?xi4>
    vector.store %arg0, %0[%arg3, %arg4] : memref<?x?xi4>, vector<8xi4>
    return
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 2)>
//      CHECK: func @vector_store_i4_dynamic
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: vector<8xi4>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//      CHECK: %[[SIZE:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK: %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi8>
//      CHECK: %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG3]], %[[ARG2]], %[[ARG4]]]
//      CHECK: %[[VEC_I8:.+]] = vector.bitcast %[[ARG0]] : vector<8xi4> to vector<4xi8>
//      CHECK: vector.store %[[VEC_I8:.+]], %[[ALLOC:.+]][%[[INDEX:.+]]] : memref<?xi8>, vector<4xi8>

//  CHECK32-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 8)>
//      CHECK32: func @vector_store_i4_dynamic
// CHECK32-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: vector<8xi4>
// CHECK32-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//      CHECK32: %[[SIZE:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK32: %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi32>
//      CHECK32: %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG3]], %[[ARG2]], %[[ARG4]]]
//      CHECK32: %[[VEC_I8:.+]] = vector.bitcast %[[ARG0]] : vector<8xi4> to vector<1xi32>
//      CHECK32: vector.store %[[VEC_I8:.+]], %[[ALLOC:.+]][%[[INDEX:.+]]] : memref<?xi32>, vector<1xi32>

// -----

func.func @vector_maskedstore_i8(%arg0: index, %arg1: index, %arg2: index, %value: vector<8xi8>) {
  %0 = memref.alloc() : memref<3x8xi8>
  %mask = vector.create_mask %arg2 : vector<8xi1>
  vector.maskedstore %0[%arg0, %arg1], %mask, %value : memref<3x8xi8>, vector<8xi1>, vector<8xi8>
  return
}
// Expect no conversions, i8 is supported.
//      CHECK: func @vector_maskedstore_i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[VAL:[a-zA-Z0-9]+]]
// CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<3x8xi8>
// CHECK-NEXT:   %[[MASK:.+]] = vector.create_mask %[[ARG2]] : vector<8xi1>
// CHECK-NEXT:   vector.maskedstore %[[ALLOC]][%[[ARG0]], %[[ARG1]]], %[[MASK]], %[[VAL]]
// CHECK-NEXT:   return

// CHECK32-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 2 + s1 floordiv 4)>
// CHECK32-DAG: #[[MASK_IDX_MAP:.+]] = affine_map<()[s0] -> ((s0 + 3) floordiv 4)>
// CHECK32:     func @vector_maskedstore_i8(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK32-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9]+]]
// CHECK32-SAME:     %[[VAL:[a-zA-Z0-9]+]]
// CHECK32:        %[[ALLOC:.+]] = memref.alloc() : memref<6xi32>
// CHECK32:        %[[ORIG_MASK:.+]] = vector.create_mask %[[ARG2]] : vector<8xi1>
// CHECK32:        %[[LIDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
// CHECK32:        %[[MASK_IDX:.+]] = affine.apply #[[MASK_IDX_MAP]]()[%[[ARG2]]]
// CHECK32:        %[[NEW_MASK:.+]] = vector.create_mask %[[MASK_IDX]] : vector<2xi1>
// CHECK32:        %[[PASS_THRU:.+]] = arith.constant dense<0> : vector<2xi32>
// CHECK32:        %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LIDX]]], %[[NEW_MASK]], %[[PASS_THRU]]
// CHECK32:        %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<2xi32> to vector<8xi8>
// CHECK32:        %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[VAL]], %[[BITCAST]] : vector<8xi1>, vector<8xi8>
// CHECK32:        %[[NEW_VAL:.+]] = vector.bitcast %[[SELECT]] : vector<8xi8> to vector<2xi32>
// CHECK32:        vector.maskedstore %[[ALLOC]][%[[LIDX]]], %[[NEW_MASK]], %[[NEW_VAL]]

// -----

func.func @vector_cst_maskedstore_i8(%arg0: index, %arg1: index, %value: vector<8xi8>) {
  %0 = memref.alloc() : memref<3x8xi8>
  %mask = vector.constant_mask [4] : vector<8xi1>
  vector.maskedstore %0[%arg0, %arg1], %mask, %value : memref<3x8xi8>, vector<8xi1>, vector<8xi8>
  return
}
// Expect no conversions, i8 is supported.
//      CHECK: func @vector_cst_maskedstore_i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[VAL:[a-zA-Z0-9]+]]
// CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<3x8xi8>
// CHECK-NEXT:   %[[MASK:.+]] = vector.constant_mask [4] : vector<8xi1>
// CHECK-NEXT:   vector.maskedstore %[[ALLOC]][%[[ARG0]], %[[ARG1]]], %[[MASK]], %[[VAL]]
// CHECK-NEXT:   return

// CHECK32-DAG: #[[LOAD_IDX_MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 2 + s1 floordiv 4)>
// CHECK32:     func @vector_cst_maskedstore_i8(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK32-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
// CHECK32-SAME:     %[[VAL:[a-zA-Z0-9]+]]
// CHECK32:        %[[ALLOC:.+]] = memref.alloc() : memref<6xi32>
// CHECK32:        %[[ORIG_MASK:.+]] = vector.constant_mask [4] : vector<8xi1>
// CHECK32:        %[[LIDX:.+]] = affine.apply #[[LOAD_IDX_MAP]]()[%[[ARG0]], %[[ARG1]]]
// CHECK32:        %[[NEW_MASK:.+]] = vector.constant_mask [1] : vector<2xi1>
// CHECK32:        %[[PASS_THRU:.+]] = arith.constant dense<0> : vector<2xi32>
// CHECK32:        %[[LOAD:.+]] = vector.maskedload %[[ALLOC]][%[[LIDX]]], %[[NEW_MASK]], %[[PASS_THRU]]
// CHECK32:        %[[BITCAST:.+]] = vector.bitcast %[[LOAD]] : vector<2xi32> to vector<8xi8>
// CHECK32:        %[[SELECT:.+]] = arith.select %[[ORIG_MASK]], %[[VAL]], %[[BITCAST]] : vector<8xi1>, vector<8xi8>
// CHECK32:        %[[NEW_VAL:.+]] = vector.bitcast %[[SELECT]] : vector<8xi8> to vector<2xi32>
// CHECK32:        vector.maskedstore %[[ALLOC]][%[[LIDX]]], %[[NEW_MASK]], %[[NEW_VAL]]
