// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --cse --split-input-file %s | FileCheck %s
// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=32" --cse --split-input-file %s | FileCheck %s --check-prefix=CHECK32

// Expect no conversions.
func.func @memref_i8() -> i8 {
    %c3 = arith.constant 3 : index
    %m = memref.alloc() : memref<4xi8, 1>
    %v = memref.load %m[%c3] : memref<4xi8, 1>
    return %v : i8
}
// CHECK-LABEL: func @memref_i8()
//       CHECK:   %[[M:.+]] = memref.alloc() : memref<4xi8, 1>
//  CHECK-NEXT:   %[[V:.+]] = memref.load %[[M]][%{{.+}}] : memref<4xi8, 1>
//  CHECK-NEXT:   return %[[V]]

// CHECK32-LABEL: func @memref_i8()
//       CHECK32:   %[[M:.+]] = memref.alloc() : memref<1xi32, 1>
//       CHECK32:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK32:   %[[V:.+]] = memref.load %[[M]][%[[C0]]] : memref<1xi32, 1>
//       CHECK32:   %[[C24:.+]] = arith.constant 24 : index
//       CHECK32:   %[[CAST:.+]] = arith.index_cast %[[C24]] : index to i32
//       CHECK32:   %[[SHIFTRT:.+]] = arith.shrsi %[[V]], %[[CAST]]
//       CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i32 to i8
//  CHECK32-NEXT:   return %[[TRUNC]]

// -----

func.func @memref_load_i4(%arg0: index) -> i4 {
    %0 = memref.alloc() : memref<5xi4>
    %1 = memref.load %0[%arg0] : memref<5xi4>
    return %1 : i4
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 2) * 8)
//      CHECK: func @memref_load_i4(
// CHECK-SAME:     %[[ARG0:.+]]: index
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:   %[[LOADVAL:.+]] = memref.load %[[ALLOC]][%[[INDEX]]]
//      CHECK:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//      CHECK:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOADVAL]], %[[CAST]]
//      CHECK:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i8 to i4
//      CHECK:   return %[[TRUNC]]

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)
//      CHECK32: func @memref_load_i4(
// CHECK32-SAME:     %[[ARG0:.+]]: index
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<1xi32>
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK32:   %[[LOADVAL:.+]] = memref.load %[[ALLOC]][%[[INDEX]]]
//      CHECK32:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//      CHECK32:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//      CHECK32:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOADVAL]], %[[CAST]]
//      CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i32 to i4
//      CHECK32:   return %[[TRUNC]]

// -----

func.func @memref_load_i4_rank2(%arg0: index, %arg1: index) -> i4 {
    %0 = memref.alloc() : memref<3x125xi4>
    memref.assume_alignment %0, 64 : memref<3x125xi4>
    %1 = memref.load %0[%arg0,%arg1] : memref<3x125xi4>
    return %1 : i4
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * 125 + s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 500 + s1 * 4 - ((s0 * 125 + s1) floordiv 2) * 8)
//      CHECK: func @memref_load_i4_rank2(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<188xi8>
//      CHECK:   memref.assume_alignment %[[ALLOC]], 64 : memref<188xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX]]]
//      CHECK:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//      CHECK:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOAD]], %[[CAST]]
//      CHECK:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i8 to i4
//      CHECK:   return %[[TRUNC]]

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * 125 + s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 500 + s1 * 4 - ((s0 * 125 + s1) floordiv 8) * 32)
//      CHECK32: func @memref_load_i4_rank2(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK32-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<47xi32>
//      CHECK32:   memref.assume_alignment %[[ALLOC]], 64 : memref<47xi32>
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX]]]
//      CHECK32:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//      CHECK32:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOAD]], %[[CAST]]
//      CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i32 to i4
//      CHECK32:   return %[[TRUNC]]

// -----

func.func @memref_load_i4_dynamic(%arg0: index, %arg1 : index, %arg2 : index, %arg3 : index) -> i4 {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xi4>
  %1 = memref.load %0[%arg2, %arg3] : memref<?x?xi4>
  return %1 : i4
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * 4 + s2 * 4 - ((s2 + s0 * s1) floordiv 2) * 8)>
//      CHECK: func @memref_load_i4_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]])
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX]]]
//      CHECK:   %[[BITOFFSET:.+]] = affine.apply #[[MAP2]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//      CHECK:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOAD]], %[[CAST]]
//      CHECK:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i8 to i4
//      CHECK:   return %[[TRUNC]]

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * 4 + s2 * 4 - ((s2 + s0 * s1) floordiv 8) * 32)>
//      CHECK32: func @memref_load_i4_dynamic(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//      CHECK32:   %[[SIZE:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//      CHECK32:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]])
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//      CHECK32:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX]]]
//      CHECK32:   %[[BITOFFSET:.+]] = affine.apply #[[MAP2]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//      CHECK32:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//      CHECK32:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOAD]], %[[CAST]]
//      CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i32 to i4
//      CHECK32:   return %[[TRUNC]]

// -----

func.func @rank_zero_memref() -> i4 {
  %0 = memref.alloc() : memref<i4>
  %1 = memref.load %0[] : memref<i4>
  return %1 : i4
}
// CHECK-LABEL: func @rank_zero_memref()
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<i8>
//       CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOC]][] : memref<i8>
//       CHECK:   %[[TRUNC:.+]] = arith.trunci %[[LOAD]] : i8 to i4
//       CHECK:   return %[[TRUNC]]

// CHECK32-LABEL: func @rank_zero_memref()
//       CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<i32>
//       CHECK32:   %[[LOAD:.+]] = memref.load %[[ALLOC]][] : memref<i32>
//       CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[LOAD]] : i32 to i4
//       CHECK32:   return %[[TRUNC]]

// -----

func.func @memref_strided_i4(%idx : index) -> i4 {
  %arr = memref.alloc() : memref<128xi4>
  %subview = memref.subview %arr[32] [32] [1] : memref<128xi4> to memref<32xi4, strided<[1], offset:32>>
  %1 = memref.load %subview[%idx] : memref<32xi4, strided<[1], offset:32>>
  return %1 : i4
}

// CHECK-LABEL: func @memref_strided_i4
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<64xi8>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ALLOC]][16] [16] [1] : memref<64xi8> to memref<16xi8, strided<[1], offset: 16>>
//       CHECK:   %[[LOAD:.+]] = memref.load %[[SUBVIEW]]

// CHECK32-LABEL: func @memref_strided_i4
//       CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<16xi32>
//       CHECK32:   %[[SUBVIEW:.+]] = memref.subview %[[ALLOC]][4] [4] [1] : memref<16xi32> to memref<4xi32, strided<[1], offset: 4>>
//       CHECK32:   %[[LOAD:.+]] = memref.load %[[SUBVIEW]]

// -----

func.func @reinterpret_cast_memref_load_0D() -> i4 {
    %0 = memref.alloc() : memref<5xi4>
    %reinterpret_cast_0 = memref.reinterpret_cast %0 to offset: [0], sizes: [], strides: [] : memref<5xi4> to memref<i4>
    %1 = memref.load %reinterpret_cast_0[] : memref<i4>
    return %1 : i4
}
// CHECK-LABEL: func @reinterpret_cast_memref_load_0D()
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<3xi8>
//       CHECK:   %[[RE_CAST:.+]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [], strides: [] : memref<3xi8> to memref<i8>
//       CHECK:   %[[LOAD:.+]] = memref.load %[[RE_CAST]][] : memref<i8>
//       CHECK:   %[[TRUNC:.+]] = arith.trunci %[[LOAD]] : i8 to i4
//       CHECK:   return %[[TRUNC]]

// CHECK32-LABEL: func @reinterpret_cast_memref_load_0D()
//       CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<1xi32>
//       CHECK32:   %[[RE_CAST:.+]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [], strides: [] : memref<1xi32> to memref<i32>
//       CHECK32:   %[[LOAD:.+]] = memref.load %[[RE_CAST]][] : memref<i32>
//       CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[LOAD]] : i32 to i4
//       CHECK32:   return %[[TRUNC]]

// -----

func.func @reinterpret_cast_memref_load_1D(%arg0: index) -> i4 {
    %0 = memref.alloc() : memref<5x5xi4>
    %reinterpret_cast_0 = memref.reinterpret_cast %0 to offset: [8], sizes: [25], strides: [1] : memref<5x5xi4> to memref<25xi4, strided<[1], offset:8>>
    %1 = memref.load %reinterpret_cast_0[%arg0] : memref<25xi4, strided<[1], offset:8>>
    return %1 : i4
}
//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 2) * 8)>
//       CHECK: func @reinterpret_cast_memref_load_1D(
//  CHECK-SAME: %[[ARG0:.+]]: index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<13xi8>
//       CHECK:   %[[RE_CAST:.+]] = memref.reinterpret_cast %[[ALLOC]] to offset: [4], sizes: [13], strides: [1] : memref<13xi8> to memref<13xi8, strided<[1], offset: 4>>
//       CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//       CHECK:   %[[LOAD:.+]] = memref.load %[[RE_CAST]][%[[INDEX]]] : memref<13xi8, strided<[1], offset: 4>>
//       CHECK:   %[[OFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//       CHECK:   %[[CAST:.+]] = arith.index_cast %[[OFFSET]] : index to i8
//       CHECK:   %[[SHR:.+]] = arith.shrsi %[[LOAD]], %[[CAST]] : i8
//       CHECK:   %[[TRUNC:.+]] = arith.trunci %[[SHR]] : i8 to i4
//       CHECK:   return %[[TRUNC]]

//   CHECK32-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
//       CHECK32: func @reinterpret_cast_memref_load_1D(
//  CHECK32-SAME: %[[ARG0:.+]]: index
//       CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<4xi32>
//       CHECK32:   %[[RE_CAST:.+]] = memref.reinterpret_cast %[[ALLOC]] to offset: [1], sizes: [4], strides: [1] : memref<4xi32> to memref<4xi32, strided<[1], offset: 1>>
//       CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//       CHECK32:   %[[LOAD:.+]] = memref.load %[[RE_CAST]][%[[INDEX]]] : memref<4xi32, strided<[1], offset: 1>>
//       CHECK32:   %[[OFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//       CHECK32:   %[[CAST:.+]] = arith.index_cast %[[OFFSET]] : index to i32
//       CHECK32:   %[[SHR:.+]] = arith.shrsi %[[LOAD]], %[[CAST]] : i32
//       CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHR]] : i32 to i4
//       CHECK32:   return %[[TRUNC]]
