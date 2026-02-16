// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 disable-atomic-rmw=true" --cse --split-input-file %s | FileCheck %s

func.func @memref_store_i4(%arg0: index, %arg1: i4) -> () {
    %0 = memref.alloc() : memref<5xi4>
    memref.store %arg1, %0[%arg0] : memref<5xi4>
    return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 2) * 8)>
//      CHECK: func @memref_store_i4(
// CHECK-SAME:     %[[ARG0:.+]]: index, %[[ARG1:.+]]: i4
//  CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
//  CHECK-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG1]] : i4 to i8
//  CHECK-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//  CHECK-DAG:   %[[BITOFFSET_I8:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//  CHECK-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i8
//  CHECK-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I8]] : i8
//  CHECK-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i8
//  CHECK-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i8
//  CHECK-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I8]] : i8
//      CHECK:   %[[ORIG:.+]] = memref.load %[[ALLOC]][%[[INDEX]]] : memref<3xi8>
//      CHECK:   %[[CLEARED:.+]] = arith.andi %[[ORIG]], %[[MASK]] : i8
//      CHECK:   %[[INSERTED:.+]] = arith.ori %[[CLEARED]], %[[SHIFTED_VAL]] : i8
//      CHECK:   memref.store %[[INSERTED]], %[[ALLOC]][%[[INDEX]]] : memref<3xi8>
//      CHECK:   return

// -----

func.func @memref_store_i4_rank2(%arg0: index, %arg1: index, %arg2: i4) -> () {
    %0 = memref.alloc() : memref<3x125xi4>
    memref.store %arg2, %0[%arg0,%arg1] : memref<3x125xi4>
    return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * 125 + s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 500 + s1 * 4 - ((s0 * 125 + s1) floordiv 2) * 8)>
//      CHECK: func @memref_store_i4_rank2(
// CHECK-SAME:     %[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i4
//  CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<188xi8>
//  CHECK-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG2]] : i4 to i8
//  CHECK-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK-DAG:   %[[BITOFFSET_I8:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//  CHECK-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i8
//  CHECK-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I8]] : i8
//  CHECK-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i8
//  CHECK-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i8
//  CHECK-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I8]] : i8
//      CHECK:   %[[ORIG:.+]] = memref.load %[[ALLOC]][%[[INDEX]]] : memref<188xi8>
//      CHECK:   %[[CLEARED:.+]] = arith.andi %[[ORIG]], %[[MASK]] : i8
//      CHECK:   %[[INSERTED:.+]] = arith.ori %[[CLEARED]], %[[SHIFTED_VAL]] : i8
//      CHECK:   memref.store %[[INSERTED]], %[[ALLOC]][%[[INDEX]]] : memref<188xi8>
//      CHECK:   return

// -----

func.func @memref_store_i4_dynamic(%arg0: index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4: i4) -> () {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xi4>
  memref.store %arg4, %0[%arg2, %arg3] : memref<?x?xi4>
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2, s0 floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * 4 + s2 * 4 - ((s2 + s0 * s1) floordiv 2) * 8)>
//      CHECK: func @memref_store_i4_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: i4
//  CHECK-DAG:   %[[SIZE:.+]] = affine.max #[[MAP0]]()[%[[ARG1]], %[[ARG0]]]
//  CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi8>
//  CHECK-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG4]] : i4 to i8
//  CHECK-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//  CHECK-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP2]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//  CHECK-DAG:   %[[BITOFFSET_I8:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//  CHECK-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i8
//  CHECK-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I8]] : i8
//  CHECK-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i8
//  CHECK-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i8
//  CHECK-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I8]] : i8
//      CHECK:   %[[ORIG:.+]] = memref.load %[[ALLOC]][%[[INDEX]]] : memref<?xi8>
//      CHECK:   %[[CLEARED:.+]] = arith.andi %[[ORIG]], %[[MASK]] : i8
//      CHECK:   %[[INSERTED:.+]] = arith.ori %[[CLEARED]], %[[SHIFTED_VAL]] : i8
//      CHECK:   memref.store %[[INSERTED]], %[[ALLOC]][%[[INDEX]]] : memref<?xi8>
//      CHECK:   return

// -----

func.func @memref_store_f4(%arg0: f4E2M1FN) -> () {
    %0 = memref.alloc() : memref<f4E2M1FN>
    memref.store %arg0, %0[] : memref<f4E2M1FN>
    return
}
// CHECK-LABEL: func @memref_store_f4(
//  CHECK-SAME:     %[[ARG0:.+]]: f4E2M1FN
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<i8>
//       CHECK:   %[[BC:.+]] = arith.bitcast %[[ARG0]] : f4E2M1FN to i4
//       CHECK:   %[[EXTUI:.+]] = arith.extui %[[BC]] : i4 to i8
//       CHECK:   memref.store %[[EXTUI]], %[[ALLOC]][] : memref<i8>
//       CHECK:   return

// -----

// 0-rank memrefs don't need RMW since they store the entire element.
func.func @rank_zero_memref_store(%arg0: i4) -> () {
  %0 = memref.alloc() : memref<i4>
  memref.store %arg0, %0[] : memref<i4>
  return
}
// CHECK-LABEL: func @rank_zero_memref
//  CHECK-SAME:     %[[ARG0:.+]]: i4
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<i8>
//       CHECK:   %[[EXTUI:.+]] = arith.extui %[[ARG0]] : i4 to i8
//       CHECK:   memref.store %[[EXTUI]], %[[ALLOC]][] : memref<i8>
//       CHECK:   return
