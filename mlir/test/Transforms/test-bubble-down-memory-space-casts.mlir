// RUN: mlir-opt %s --bubble-down-memory-space-casts | FileCheck %s

#map = affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>
// CHECK-LABEL:   func.func @load_store(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[ARG1]]] : memref<?xf32, 1>
// CHECK:           memref.store %[[VAL_0]], %[[ARG0]]{{\[}}%[[ARG1]]] : memref<?xf32, 1>
// CHECK:           return
// CHECK:         }
func.func @load_store(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %0 = memref.load %memspacecast[%arg1] : memref<?xf32>
  memref.store %0, %memspacecast[%arg1] : memref<?xf32>
  return
}

// CHECK-LABEL:   func.func @load_store_unfoldable(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = memref.memory_space_cast %[[ARG0]] : memref<?xf32, 1> to memref<?xf32, 2>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]]{{\[}}%[[ARG1]]] : memref<?xf32, 2>
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]]{{\[}}%[[ARG1]]] : memref<?xf32, 2>
// CHECK:           return
// CHECK:         }
func.func @load_store_unfoldable(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32, 2>
  %0 = memref.load %memspacecast[%arg1] : memref<?xf32, 2>
  memref.store %0, %memspacecast[%arg1] : memref<?xf32, 2>
  return
}

// CHECK-LABEL:   func.func @cast(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<2xf32, 1>,
// CHECK-SAME:                    %[[ARG1:.*]]: memref<*xf32, 1>) -> (memref<*xf32>, memref<3x2xf32>) {
// CHECK:           %[[VAL_0:.*]] = memref.cast %[[ARG0]] : memref<2xf32, 1> to memref<*xf32, 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<*xf32, 1> to memref<*xf32>
// CHECK:           %[[VAL_2:.*]] = memref.cast %[[ARG1]] : memref<*xf32, 1> to memref<3x2xf32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.memory_space_cast %[[VAL_2]] : memref<3x2xf32, 1> to memref<3x2xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_3]] : memref<*xf32>, memref<3x2xf32>
// CHECK:         }
func.func @cast(%arg0: memref<2xf32, 1>, %arg1: memref<*xf32, 1>) -> (memref<*xf32>, memref<3x2xf32>) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<2xf32, 1> to memref<2xf32>
  %1 = memref.cast %memspacecast : memref<2xf32> to memref<*xf32>
  %memspacecast_1 = memref.memory_space_cast %arg1 : memref<*xf32, 1> to memref<*xf32>
  %2 = memref.cast %memspacecast_1 : memref<*xf32> to memref<3x2xf32>
  return %1, %2 : memref<*xf32>, memref<3x2xf32>
}

// CHECK-LABEL:   func.func @view(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<?xi8, 1>,
// CHECK-SAME:                    %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> memref<?x?xi8> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_1:.*]] = memref.view %[[ARG0]]{{\[}}%[[ARG1]]]{{\[}}%[[ARG2]], %[[VAL_0]]] : memref<?xi8, 1> to memref<?x?xi8, 1>
// CHECK:           %[[VAL_2:.*]] = memref.memory_space_cast %[[VAL_1]] : memref<?x?xi8, 1> to memref<?x?xi8>
// CHECK:           return %[[VAL_2]] : memref<?x?xi8>
// CHECK:         }
func.func @view(%arg0: memref<?xi8, 1>, %arg1: index, %arg2: index) -> memref<?x?xi8> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xi8, 1> to memref<?xi8>
  %c100 = arith.constant 100 : index
  %view = memref.view %memspacecast[%arg1][%arg2, %c100] : memref<?xi8> to memref<?x?xi8>
  return %view : memref<?x?xi8>
}

// CHECK-LABEL:   func.func @subview(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?x?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) -> memref<8x2xf32, strided<[?, 2], offset: ?>> {
// CHECK:           %[[VAL_0:.*]] = memref.subview %[[ARG0]][4, 2] [8, 2] [3, 2] : memref<?x?xf32, 1> to memref<8x2xf32, strided<[?, 2], offset: ?>, 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<8x2xf32, strided<[?, 2], offset: ?>, 1> to memref<8x2xf32, strided<[?, 2], offset: ?>>
// CHECK:           return %[[VAL_1]] : memref<8x2xf32, strided<[?, 2], offset: ?>>
// CHECK:         }
func.func @subview(%arg0: memref<?x?xf32, 1>, %arg1: index) -> memref<8x2xf32, strided<[?, 2], offset: ?>> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?x?xf32, 1> to memref<?x?xf32>
  %subview = memref.subview %memspacecast[4, 2] [8, 2] [3, 2] : memref<?x?xf32> to memref<8x2xf32, strided<[?, 2], offset: ?>>
  return %subview : memref<8x2xf32, strided<[?, 2], offset: ?>>
}

// CHECK-LABEL:   func.func @reinterpret_cast(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[VAL_1]]], sizes: [10, %[[VAL_0]]], strides: {{\[}}%[[VAL_0]], 1] : memref<?xf32, 1> to memref<10x?xf32, strided<[?, 1], offset: ?>, 1>
// CHECK:           %[[VAL_3:.*]] = memref.memory_space_cast %[[VAL_2]] : memref<10x?xf32, strided<[?, 1], offset: ?>, 1> to memref<10x?xf32, strided<[?, 1], offset: ?>>
// CHECK:           return %[[VAL_3]] : memref<10x?xf32, strided<[?, 1], offset: ?>>
// CHECK:         }
func.func @reinterpret_cast(%arg0: memref<?xf32, 1>, %arg1: index) -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%c0], sizes: [10, %c10], strides: [%c10, 1] : memref<?xf32> to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %reinterpret_cast : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// CHECK-LABEL:   func.func @reshape(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?x?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<1xindex>) -> memref<?xf32> {
// CHECK:           %[[VAL_0:.*]] = memref.reshape %[[ARG0]](%[[ARG1]]) : (memref<?x?xf32, 1>, memref<1xindex>) -> memref<?xf32, 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<?xf32, 1> to memref<?xf32>
// CHECK:           return %[[VAL_1]] : memref<?xf32>
// CHECK:         }
func.func @reshape(%arg0: memref<?x?xf32, 1>, %arg1: memref<1xindex>) -> memref<?xf32> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?x?xf32, 1> to memref<?x?xf32>
  %reshape = memref.reshape %memspacecast(%arg1) : (memref<?x?xf32>, memref<1xindex>) -> memref<?xf32>
  return %reshape : memref<?xf32>
}

// CHECK-LABEL:   func.func @expand_shape(
// CHECK-SAME:      %[[ARG0:.*]]: memref<12xf32, 1>) -> memref<3x4xf32> {
// CHECK:           %[[VAL_0:.*]] = memref.expand_shape %[[ARG0]] {{\[\[}}0, 1]] output_shape [3, 4] : memref<12xf32, 1> into memref<3x4xf32, 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<3x4xf32, 1> to memref<3x4xf32>
// CHECK:           return %[[VAL_1]] : memref<3x4xf32>
// CHECK:         }
func.func @expand_shape(%arg0: memref<12xf32, 1>) -> memref<3x4xf32> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<12xf32, 1> to memref<12xf32>
  %expand_shape = memref.expand_shape %memspacecast [[0, 1]] output_shape [3, 4] : memref<12xf32> into memref<3x4xf32>
  return %expand_shape : memref<3x4xf32>
}

// CHECK-LABEL:   func.func @collapse_shape(
// CHECK-SAME:      %[[ARG0:.*]]: memref<3x4xf32, 1>) -> memref<12xf32> {
// CHECK:           %[[VAL_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[\[}}0, 1]] : memref<3x4xf32, 1> into memref<12xf32, 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<12xf32, 1> to memref<12xf32>
// CHECK:           return %[[VAL_1]] : memref<12xf32>
// CHECK:         }
func.func @collapse_shape(%arg0: memref<3x4xf32, 1>) -> memref<12xf32> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<3x4xf32, 1> to memref<3x4xf32>
  %collapse_shape = memref.collapse_shape %memspacecast [[0, 1]] : memref<3x4xf32> into memref<12xf32>
  return %collapse_shape : memref<12xf32>
}

// CHECK-LABEL:   func.func @transpose(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?x?xf32, 1>) -> memref<?x?xf32, #[[$ATTR_0]]> {
// CHECK:           %[[VAL_0:.*]] = memref.transpose %[[ARG0]] (d0, d1) -> (d1, d0) : memref<?x?xf32, 1> to memref<?x?xf32, #[[$ATTR_0]], 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<?x?xf32, #[[$ATTR_0]], 1> to memref<?x?xf32, #[[$ATTR_0]]>
// CHECK:           return %[[VAL_1]] : memref<?x?xf32, #[[$ATTR_0]]>
// CHECK:         }
func.func @transpose(%arg0: memref<?x?xf32, 1>) -> memref<?x?xf32, #map> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?x?xf32, 1> to memref<?x?xf32>
  %transpose = memref.transpose %memspacecast (d0, d1) -> (d1, d0) : memref<?x?xf32> to memref<?x?xf32, #map>
  return %transpose : memref<?x?xf32, #map>
}

// CHECK-LABEL:   func.func @atomic_rmw(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_0:.*]] = memref.atomic_rmw addf %[[ARG2]], %[[ARG0]]{{\[}}%[[ARG1]]] : (f32, memref<?xf32, 1>) -> f32
// CHECK:           return %[[VAL_0]] : f32
// CHECK:         }
func.func @atomic_rmw(%arg0: memref<?xf32, 1>, %arg1: index, %arg2: f32) -> f32 {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %0 = memref.atomic_rmw addf %arg2, %memspacecast[%arg1] : (f32, memref<?xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @assume_alignment(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>) -> memref<?xf32> {
// CHECK:           %[[VAL_0:.*]] = memref.assume_alignment %[[ARG0]], 16 : memref<?xf32, 1>
// CHECK:           %[[VAL_1:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<?xf32, 1> to memref<?xf32>
// CHECK:           return %[[VAL_1]] : memref<?xf32>
// CHECK:         }
func.func @assume_alignment(%arg0: memref<?xf32, 1>) -> memref<?xf32> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %1 = memref.assume_alignment %memspacecast, 16 : memref<?xf32>
  return %1 : memref<?xf32>
}

// CHECK-LABEL:   func.func @op_with_cast_sequence(
// CHECK-SAME:      %[[ARG0:.*]]: memref<4x4xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: f32) -> memref<16xf32> {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = memref.expand_shape %[[ARG0]] {{\[\[}}0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32, 1> into memref<4x2x2xf32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0, 1, 2]] : memref<4x2x2xf32, 1> into memref<16xf32, 1>
// CHECK:           %[[VAL_4:.*]] = memref.memory_space_cast %[[VAL_3]] : memref<16xf32, 1> to memref<16xf32>
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<16xf32, 1>
// CHECK:           %[[VAL_6:.*]] = arith.addf %[[VAL_5]], %[[ARG2]] : f32
// CHECK:           memref.store %[[VAL_6]], %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<16xf32, 1>
// CHECK:           %[[VAL_7:.*]] = memref.atomic_rmw addf %[[ARG2]], %[[VAL_3]]{{\[}}%[[VAL_0]]] : (f32, memref<16xf32, 1>) -> f32
// CHECK:           return %[[VAL_4]] : memref<16xf32>
// CHECK:         }
func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<4x4xf32, 1> to memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %expanded = memref.expand_shape %memspacecast [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32> into memref<4x2x2xf32>
  %collapsed = memref.collapse_shape %expanded [[0, 1, 2]] : memref<4x2x2xf32> into memref<16xf32>
  %loaded = memref.load %collapsed[%c0] : memref<16xf32>
  %added = arith.addf %loaded, %arg2 : f32
  memref.store %added, %collapsed[%c0] : memref<16xf32>
  %atomic_result = memref.atomic_rmw addf %arg2, %collapsed[%c4] : (f32, memref<16xf32>) -> f32
  return %collapsed : memref<16xf32>
}

// CHECK-LABEL:   func.func @transfer_read_write(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = vector.transfer_read %[[ARG0]]{{\[}}%[[ARG1]]], %[[VAL_0]] : memref<?xf32, 1>, vector<4xf32>
// CHECK:           vector.transfer_write %[[VAL_1]], %[[ARG0]]{{\[}}%[[ARG1]]] : vector<4xf32>, memref<?xf32, 1>
// CHECK:           return
// CHECK:         }
func.func @transfer_read_write(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %memspacecast[%arg1], %c0 : memref<?xf32>, vector<4xf32>
  vector.transfer_write %0, %memspacecast[%arg1] : vector<4xf32>, memref<?xf32>
  return
}

// NOTE: The operations disappear because they can get folded.
// CHECK-LABEL:   func.func @transfer_read_write_tensor(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: index) -> tensor<?xf32> {
// CHECK:           return %[[ARG0]] : tensor<?xf32>
// CHECK:         }
func.func @transfer_read_write_tensor(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%arg1], %c0 : tensor<?xf32>, vector<4xf32>
  %1 = vector.transfer_write %0, %arg0[%arg1] : vector<4xf32>, tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL:   func.func @vector_load_store(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = vector.load %[[ARG0]]{{\[}}%[[ARG1]]] : memref<?xf32, 1>, vector<4xf32>
// CHECK:           vector.store %[[VAL_0]], %[[ARG0]]{{\[}}%[[ARG1]]] : memref<?xf32, 1>, vector<4xf32>
// CHECK:           return
// CHECK:         }
func.func @vector_load_store(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %0 = vector.load %memspacecast[%arg1] : memref<?xf32>, vector<4xf32>
  vector.store %0, %memspacecast[%arg1] : memref<?xf32>, vector<4xf32>
  return
}

// CHECK-LABEL:   func.func @masked_load_store(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant dense<[true, true, false, false]> : vector<4xi1>
// CHECK:           %[[VAL_2:.*]] = vector.maskedload %[[ARG0]]{{\[}}%[[ARG1]]], %[[VAL_1]], %[[VAL_0]] : memref<?xf32, 1>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:           vector.maskedstore %[[ARG0]]{{\[}}%[[ARG1]]], %[[VAL_1]], %[[VAL_2]] : memref<?xf32, 1>, vector<4xi1>, vector<4xf32>
// CHECK:           return
// CHECK:         }
func.func @masked_load_store(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %mask = arith.constant dense<[true, true, false, false]> : vector<4xi1>
  %passthrough = arith.constant dense<0.0> : vector<4xf32>
  %0 = vector.maskedload %memspacecast[%arg1], %mask, %passthrough : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  vector.maskedstore %memspacecast[%arg1], %mask, %0 : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL:   func.func @gather_scatter(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant dense<true> : vector<4xi1>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = vector.gather %[[ARG0]]{{\[}}%[[VAL_3]]] {{\[}}%[[VAL_2]]], %[[VAL_1]], %[[VAL_0]] : memref<?xf32, 1>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:           vector.scatter %[[ARG0]]{{\[}}%[[VAL_3]]] {{\[}}%[[VAL_2]]], %[[VAL_1]], %[[VAL_4]] : memref<?xf32, 1>, vector<4xindex>, vector<4xi1>, vector<4xf32>
// CHECK:           return
// CHECK:         }
func.func @gather_scatter(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
  %mask = arith.constant dense<true> : vector<4xi1>
  %passthrough = arith.constant dense<0.0> : vector<4xf32>
  %0 = vector.gather %memspacecast[%c0] [%indices], %mask, %passthrough : memref<?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  vector.scatter %memspacecast[%c0] [%indices], %mask, %0 : memref<?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL:   func.func @expandload_compressstore(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant dense<[true, true, false, false]> : vector<4xi1>
// CHECK:           %[[VAL_2:.*]] = vector.expandload %[[ARG0]]{{\[}}%[[ARG1]]], %[[VAL_1]], %[[VAL_0]] : memref<?xf32, 1>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:           vector.compressstore %[[ARG0]]{{\[}}%[[ARG1]]], %[[VAL_1]], %[[VAL_2]] : memref<?xf32, 1>, vector<4xi1>, vector<4xf32>
// CHECK:           return
// CHECK:         }
func.func @expandload_compressstore(%arg0: memref<?xf32, 1>, %arg1: index) {
  %memspacecast = memref.memory_space_cast %arg0 : memref<?xf32, 1> to memref<?xf32>
  %mask = arith.constant dense<[true, true, false, false]> : vector<4xi1>
  %passthrough = arith.constant dense<0.0> : vector<4xf32>
  %0 = vector.expandload %memspacecast[%arg1], %mask, %passthrough : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  vector.compressstore %memspacecast[%arg1], %mask, %0 : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return
}
