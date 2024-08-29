// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --cse --verify-diagnostics --split-input-file %s | FileCheck %s
// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=32" --cse --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK32

// Expect no conversions.
func.func @memref_i8() -> i8 {
    %c3 = arith.constant 3 : index
    %m = memref.alloc() : memref<4xi8, 1>
    %v = memref.load %m[%c3] : memref<4xi8, 1>
    memref.dealloc %m : memref<4xi8, 1>
    return %v : i8
}
// CHECK-LABEL: func @memref_i8()
//       CHECK:   %[[M:.+]] = memref.alloc() : memref<4xi8, 1>
//  CHECK-NEXT:   %[[V:.+]] = memref.load %[[M]][%{{.+}}] : memref<4xi8, 1>
//  CHECK-NEXT:   memref.dealloc %[[M]]
//  CHECK-NEXT:   return %[[V]]

// CHECK32-LABEL: func @memref_i8()
//       CHECK32:   %[[M:.+]] = memref.alloc() : memref<1xi32, 1>
//       CHECK32:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK32:   %[[V:.+]] = memref.load %[[M]][%[[C0]]] : memref<1xi32, 1>
//       CHECK32:   %[[C24:.+]] = arith.constant 24 : index
//       CHECK32:   %[[CAST:.+]] = arith.index_cast %[[C24]] : index to i32
//       CHECK32:   %[[SHIFTRT:.+]] = arith.shrsi %[[V]], %[[CAST]]
//       CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i32 to i8
//  CHECK32-NEXT:   memref.dealloc %[[M]]
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

func.func @memref_subview_dynamic_offset_i4(%idx : index) -> i4 {
  %c0 = arith.constant 0 : index
  %arr = memref.alloc() : memref<512x64x8x16xi4>
  %subview = memref.subview %arr[%idx, 0, 0, 0] [16, 64, 8, 16] [1, 1, 1, 1] : memref<512x64x8x16xi4>
                                                                            to memref<16x64x8x16xi4, strided<[8192, 128, 16, 1], offset: ?>>
  %ld = memref.load %subview[%c0, %c0, %c0, %c0] : memref<16x64x8x16xi4, strided<[8192, 128, 16, 1], offset: ?>>
  return %ld : i4
}

// CHECK-LABEL:   func.func @memref_subview_dynamic_offset_i4(
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<2097152xi8>
// CHECK:           %[[IDX:.*]] = affine.apply
// CHECK:           %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]][%[[IDX]]] [65536] [1] : memref<2097152xi8> to memref<65536xi8, strided<[1], offset: ?>>
// CHECK:           memref.load %[[SUBVIEW]]

// CHECK32-LABEL:   func.func @memref_subview_dynamic_offset_i4(
// CHECK32:           %[[ALLOC:.*]] = memref.alloc() : memref<524288xi32>
// CHECK32:           %[[IDX:.*]] = affine.apply
// CHECK32:           %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]][%[[IDX]]] [16384] [1] : memref<524288xi32> to memref<16384xi32, strided<[1], offset: ?>>
// CHECK32:           memref.load %[[SUBVIEW]]

// -----


func.func @negative_memref_subview_non_contiguous(%idx : index) -> i4 {
  %c0 = arith.constant 0 : index
  %arr = memref.alloc() : memref<40x40xi4>
  // expected-error @+1 {{failed to legalize operation 'memref.subview' that was explicitly marked illegal}}
  %subview = memref.subview %arr[%idx, 0] [4, 8] [1, 1] : memref<40x40xi4> to memref<4x8xi4, strided<[40, 1], offset:?>>
  %ld = memref.load %subview[%c0, %c0] : memref<4x8xi4, strided<[40, 1], offset:?>>
  return %ld : i4
}

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

// -----

func.func @memref_alloca_load_i4(%arg0: index) -> i4 {
    %0 = memref.alloca() : memref<5xi4>
    %1 = memref.load %0[%arg0] : memref<5xi4>
    return %1 : i4
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 2) * 8)
//      CHECK: func @memref_alloca_load_i4(
// CHECK-SAME:     %[[ARG0:.+]]: index
//      CHECK:   %[[ALLOCA:.+]] = memref.alloca() : memref<3xi8>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:   %[[LOADVAL:.+]] = memref.load %[[ALLOCA]][%[[INDEX]]]
//      CHECK:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//      CHECK:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOADVAL]], %[[CAST]]
//      CHECK:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i8 to i4
//      CHECK:   return %[[TRUNC]]

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)
//      CHECK32: func @memref_alloca_load_i4(
// CHECK32-SAME:     %[[ARG0:.+]]: index
//      CHECK32:   %[[ALLOCA:.+]] = memref.alloca() : memref<1xi32>
//      CHECK32:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK32:   %[[LOADVAL:.+]] = memref.load %[[ALLOCA]][%[[INDEX]]]
//      CHECK32:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//      CHECK32:   %[[CAST:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//      CHECK32:   %[[SHIFTRT:.+]] = arith.shrsi %[[LOADVAL]], %[[CAST]]
//      CHECK32:   %[[TRUNC:.+]] = arith.trunci %[[SHIFTRT]] : i32 to i4
//      CHECK32:   return %[[TRUNC]]

// -----

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
//      CHECK:   %[[CLEAR_RMW:.+]] = memref.atomic_rmw andi %[[MASK]], %[[ALLOC]][%[[INDEX]]] : (i8, memref<3xi8>) -> i8
//      CHECK:   %[[WRITE_RMW:.+]] = memref.atomic_rmw ori %[[SHIFTED_VAL]], %[[ALLOC]][%[[INDEX]]] : (i8, memref<3xi8>) -> i8
//      CHECK:   return

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
//      CHECK32: func @memref_store_i4(
// CHECK32-SAME:     %[[ARG0:.+]]: index, %[[ARG1:.+]]: i4
//  CHECK32-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<1xi32>
//  CHECK32-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG1]] : i4 to i32
//  CHECK32-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK32-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]]]
//  CHECK32-DAG:   %[[BITOFFSET_I32:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//  CHECK32-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i32
//  CHECK32-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I32]] : i32
//  CHECK32-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i32
//  CHECK32-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i32
//  CHECK32-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I32]] : i32
//      CHECK32:   %[[CLEAR_RMW:.+]] = memref.atomic_rmw andi %[[MASK]], %[[ALLOC]][%[[INDEX]]] : (i32, memref<1xi32>) -> i32
//      CHECK32:   %[[WRITE_RMW:.+]] = memref.atomic_rmw ori %[[SHIFTED_VAL]], %[[ALLOC]][%[[INDEX]]] : (i32, memref<1xi32>) -> i32
//      CHECK32:   return

// -----

func.func @memref_store_i4_rank2(%arg0: index, %arg1: index, %arg2: i4) -> () {
    %0 = memref.alloc() : memref<3x125xi4>
    memref.assume_alignment %0, 64 : memref<3x125xi4>
    memref.store %arg2, %0[%arg0,%arg1] : memref<3x125xi4>
    return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * 125 + s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 500 + s1 * 4 - ((s0 * 125 + s1) floordiv 2) * 8)>
//      CHECK: func @memref_store_i4_rank2(
// CHECK-SAME:     %[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i4
//  CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<188xi8>
//  CHECK-DAG:   memref.assume_alignment %[[ALLOC]], 64 : memref<188xi8>
//  CHECK-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG2]] : i4 to i8
//  CHECK-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK-DAG:   %[[BITOFFSET_I8:.+]] = arith.index_cast %[[BITOFFSET]] : index to i8
//  CHECK-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i8
//  CHECK-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I8]] : i8
//  CHECK-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i8
//  CHECK-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i8
//  CHECK-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I8]] : i8
//      CHECK:   %[[CLEAR_RMW:.+]] = memref.atomic_rmw andi %[[MASK]], %[[ALLOC]][%[[INDEX]]] : (i8, memref<188xi8>) -> i8
//      CHECK:   %[[WRITE_RMW:.+]] = memref.atomic_rmw ori %[[SHIFTED_VAL]], %[[ALLOC]][%[[INDEX]]] : (i8, memref<188xi8>) -> i8
//      CHECK:   return

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * 125 + s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 500 + s1 * 4 - ((s0 * 125 + s1) floordiv 8) * 32)>
//      CHECK32: func @memref_store_i4_rank2(
// CHECK32-SAME:     %[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i4
//  CHECK32-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<47xi32>
//  CHECK32-DAG:   memref.assume_alignment %[[ALLOC]], 64 : memref<47xi32>
//  CHECK32-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG2]] : i4 to i32
//  CHECK32-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK32-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP1]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK32-DAG:   %[[BITOFFSET_I32:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//  CHECK32-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i32
//  CHECK32-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I32]] : i32
//  CHECK32-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i32
//  CHECK32-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i32
//  CHECK32-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I32]] : i32
//      CHECK32:   %[[CLEAR_RMW:.+]] = memref.atomic_rmw andi %[[MASK]], %[[ALLOC]][%[[INDEX]]] : (i32, memref<47xi32>) -> i32
//      CHECK32:   %[[WRITE_RMW:.+]] = memref.atomic_rmw ori %[[SHIFTED_VAL]], %[[ALLOC]][%[[INDEX]]] : (i32, memref<47xi32>) -> i32
//      CHECK32:   return

// -----

func.func @memref_store_i4_dynamic(%arg0: index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4: i4) -> () {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xi4>
  memref.store %arg4, %0[%arg2, %arg3] : memref<?x?xi4>
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * 4 + s2 * 4 - ((s2 + s0 * s1) floordiv 2) * 8)>
//      CHECK: func @memref_store_i4_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: i4
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
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
//      CHECK:   %[[CLEAR_RMW:.+]] = memref.atomic_rmw andi %[[MASK]], %[[ALLOC]][%[[INDEX]]] : (i8, memref<?xi8>) -> i8
//      CHECK:   %[[WRITE_RMW:.+]] = memref.atomic_rmw ori %[[SHIFTED_VAL]], %[[ALLOC]][%[[INDEX]]] : (i8, memref<?xi8>) -> i8
//      CHECK:   return

//  CHECK32-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + s0 * s1) floordiv 8)>
//  CHECK32-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * 4 + s2 * 4 - ((s2 + s0 * s1) floordiv 8) * 32)>
//      CHECK32: func @memref_store_i4_dynamic(
// CHECK32-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK32-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: i4
//  CHECK32-DAG:   %[[SIZE:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]], %[[ARG1]]]
//  CHECK32-DAG:   %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xi32>
//  CHECK32-DAG:   %[[EXTUI:.+]] = arith.extui %[[ARG4]] : i4 to i32
//  CHECK32-DAG:   %[[INDEX:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//  CHECK32-DAG:   %[[BITOFFSET:.+]] = affine.apply #[[MAP2]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//  CHECK32-DAG:   %[[BITOFFSET_I32:.+]] = arith.index_cast %[[BITOFFSET]] : index to i32
//  CHECK32-DAG:   %[[MASK_BASE:.+]] = arith.constant 15 : i32
//  CHECK32-DAG:   %[[MASK_SHIFTED:.+]] = arith.shli %[[MASK_BASE]], %[[BITOFFSET_I32]] : i32
//  CHECK32-DAG:   %[[CST_NEG_ONE:.+]] = arith.constant -1 : i32
//  CHECK32-DAG:   %[[MASK:.+]] = arith.xori %[[MASK_SHIFTED]], %[[CST_NEG_ONE]] : i32
//  CHECK32-DAG:   %[[SHIFTED_VAL:.+]] = arith.shli %[[EXTUI]], %[[BITOFFSET_I32]] : i32
//      CHECK32:   %[[CLEAR_RMW:.+]] = memref.atomic_rmw andi %[[MASK]], %[[ALLOC]][%[[INDEX]]] : (i32, memref<?xi32>) -> i32
//      CHECK32:   %[[WRITE_RMW:.+]] = memref.atomic_rmw ori %[[SHIFTED_VAL]], %[[ALLOC]][%[[INDEX]]] : (i32, memref<?xi32>) -> i32
//      CHECK32:   return

// -----

func.func @rank_zero_memref_store(%arg0: i4) -> () {
  %0 = memref.alloc() : memref<i4>
  memref.store %arg0, %0[] : memref<i4>
  return
}
// CHECK-LABEL: func @rank_zero_memref
//  CHECK-SAME:     %[[ARG0:.+]]: i4
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<i8>
//       CHECK:   %[[EXTUI:.+]] = arith.extui %[[ARG0]] : i4 to i8
//       CHECK:   %[[WRITE_RMW:.+]] = memref.atomic_rmw assign %[[EXTUI]], %[[ALLOC]][] : (i8, memref<i8>) -> i8
//       CHECK:   return

// CHECK32-LABEL: func @rank_zero_memref
//  CHECK32-SAME:     %[[ARG0:.+]]: i4
//       CHECK32:   %[[ALLOC:.+]] = memref.alloc() : memref<i32>
//       CHECK32:   %[[EXTUI:.+]] = arith.extui %[[ARG0]] : i4 to i32
//       CHECK32:   %[[WRITE_RMW:.+]] = memref.atomic_rmw assign %[[EXTUI]], %[[ALLOC]][] : (i32, memref<i32>) -> i32
//       CHECK32:   return

// -----

func.func @memref_collapse_shape_i4(%idx0 : index, %idx1 : index) -> i4 {
  %arr = memref.alloc() : memref<32x8x128xi4>
  %collapse = memref.collapse_shape %arr[[0, 1], [2]] : memref<32x8x128xi4> into memref<256x128xi4>
  %1 = memref.load %collapse[%idx0, %idx1] : memref<256x128xi4>
  return %1 : i4
}

// CHECK-LABEL:   func.func @memref_collapse_shape_i4(
//       CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<16384xi8>
//   CHECK-NOT:     memref.collapse_shape
//       CHECK:     memref.load %[[ALLOC]][%{{.*}}] : memref<16384xi8>

// CHECK32-LABEL:   func.func @memref_collapse_shape_i4(
//       CHECK32:     %[[ALLOC:.*]] = memref.alloc() : memref<4096xi32>
//   CHECK32-NOT:     memref.collapse_shape
//       CHECK32:     memref.load %[[ALLOC]][%{{.*}}] : memref<4096xi32>

// -----

func.func @memref_expand_shape_i4(%idx0 : index, %idx1 : index, %idx2 : index) -> i4 {
  %arr = memref.alloc() : memref<256x128xi4>
  %expand = memref.expand_shape %arr[[0, 1], [2]] output_shape [32, 8, 128] : memref<256x128xi4> into memref<32x8x128xi4>
  %1 = memref.load %expand[%idx0, %idx1, %idx2] : memref<32x8x128xi4>
  return %1 : i4
}

// CHECK-LABEL:   func.func @memref_expand_shape_i4(
//       CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<16384xi8>
//   CHECK-NOT:     memref.expand_shape
//       CHECK:     memref.load %[[ALLOC]][%{{.*}}] : memref<16384xi8>

// CHECK32-LABEL:   func.func @memref_expand_shape_i4(
//       CHECK32:     %[[ALLOC:.*]] = memref.alloc() : memref<4096xi32>
//   CHECK32-NOT:     memref.expand_shape
//       CHECK32:     memref.load %[[ALLOC]][%{{.*}}] : memref<4096xi32>

// -----

func.func @memref_memory_space_cast_i4(%arg0: memref<32x128xi4, 1>) -> memref<32x128xi4> {
  %cast = memref.memory_space_cast %arg0 : memref<32x128xi4, 1> to memref<32x128xi4>
  return %cast : memref<32x128xi4>
}

// CHECK-LABEL:   func.func @memref_memory_space_cast_i4(
//  CHECK-SAME:   %[[ARG0:.*]]: memref<2048xi8, 1>
//       CHECK:     %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<2048xi8, 1> to memref<2048xi8>
//       CHECK:     return %[[CAST]]

// CHECK32-LABEL:   func.func @memref_memory_space_cast_i4(
//  CHECK32-SAME:   %[[ARG0:.*]]: memref<512xi32, 1>
//       CHECK32:     %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<512xi32, 1> to memref<512xi32>
//       CHECK32:     return %[[CAST]]

// -----

func.func @memref_copy_i4(%arg0: memref<32x128xi4, 1>, %arg1: memref<32x128xi4>) {
  memref.copy %arg0, %arg1 : memref<32x128xi4, 1> to memref<32x128xi4>
  return
}

// CHECK-LABEL:   func.func @memref_copy_i4(
//  CHECK-SAME:   %[[ARG0:.*]]: memref<2048xi8, 1>, %[[ARG1:.*]]: memref<2048xi8>
//       CHECK:     memref.copy %[[ARG0]], %[[ARG1]]
//       CHECK:     return

// CHECK32-LABEL:   func.func @memref_copy_i4(
//  CHECK32-SAME:   %[[ARG0:.*]]: memref<512xi32, 1>, %[[ARG1:.*]]: memref<512xi32>
//       CHECK32:     memref.copy %[[ARG0]], %[[ARG1]]
//       CHECK32:     return

// -----

!colMajor = memref<8x8xi4, strided<[1, 8]>>
func.func @copy_distinct_layouts(%idx : index) -> i4 {
  %c0 = arith.constant 0 : index
  %arr = memref.alloc() : memref<8x8xi4>
  %arr2 = memref.alloc() : !colMajor
  // expected-error @+1 {{failed to legalize operation 'memref.copy' that was explicitly marked illegal}}
  memref.copy %arr, %arr2 : memref<8x8xi4> to !colMajor
  %ld = memref.load %arr2[%c0, %c0] : !colMajor
  return %ld : i4
}
