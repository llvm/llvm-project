// RUN: mlir-opt %s -convert-vector-to-llvm='index-bitwidth=32' -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @masked_reduce_add_f32_scalable(
// CHECK-SAME:                                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<[16]xf32>,
// CHECK-SAME:                                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<[16]xi1>) -> f32 {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = "llvm.intr.vscale"() : () -> i32
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_3]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = "llvm.intr.vp.reduce.fadd"(%[[VAL_2]], %[[VAL_0]], %[[VAL_1]], %[[VAL_7]]) : (f32, vector<[16]xf32>, vector<[16]xi1>, i32) -> f32
// CHECK:           return %[[VAL_8]] : f32
// CHECK:         }
func.func @masked_reduce_add_f32_scalable(%arg0: vector<[16]xf32>, %mask : vector<[16]xi1>) -> f32 {
  %0 = vector.mask %mask { vector.reduction <add>, %arg0 : vector<[16]xf32> into f32 } : vector<[16]xi1> -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL:   func.func @shuffle_1D(
// CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<2xf32>,
// CHECK-SAME:                          %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<3xf32>) -> vector<5xf32> {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.poison : vector<5xf32>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(2 : index) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_3]] : i32] : vector<3xf32>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.insertelement %[[VAL_4]], %[[VAL_2]]{{\[}}%[[VAL_5]] : i32] : vector<5xf32>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_7]] : i32] : vector<3xf32>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.insertelement %[[VAL_8]], %[[VAL_6]]{{\[}}%[[VAL_9]] : i32] : vector<5xf32>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_11]] : i32] : vector<3xf32>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(2 : index) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.insertelement %[[VAL_12]], %[[VAL_10]]{{\[}}%[[VAL_13]] : i32] : vector<5xf32>
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_15]] : i32] : vector<2xf32>
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(3 : index) : i32
// CHECK:           %[[VAL_18:.*]] = llvm.insertelement %[[VAL_16]], %[[VAL_14]]{{\[}}%[[VAL_17]] : i32] : vector<5xf32>
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_20:.*]] = llvm.extractelement %[[VAL_0]]{{\[}}%[[VAL_19]] : i32] : vector<2xf32>
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(4 : index) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.insertelement %[[VAL_20]], %[[VAL_18]]{{\[}}%[[VAL_21]] : i32] : vector<5xf32>
// CHECK:           return %[[VAL_22]] : vector<5xf32>
// CHECK:         }
func.func @shuffle_1D(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<5xf32> {
  %1 = vector.shuffle %arg0, %arg1 [4, 3, 2, 1, 0] : vector<2xf32>, vector<3xf32>
  return %1 : vector<5xf32>
}

// -----

// CHECK-LABEL:   func.func @extractelement_from_vec_0d_f32(
// CHECK-SAME:                                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<f32>) -> f32 {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<f32> to vector<1xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_2]] : i32] : vector<1xf32>
// CHECK:           return %[[VAL_3]] : f32
// CHECK:         }
func.func @extractelement_from_vec_0d_f32(%arg0: vector<f32>) -> f32 {
  %1 = vector.extractelement %arg0[] : vector<f32>
  return %1 : f32
}

// -----

// CHECK-LABEL:   func.func @insertelement_into_vec_0d_f32(
// CHECK-SAME:                                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                                             %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<f32>) -> vector<f32> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<f32> to vector<1xf32>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.insertelement %[[VAL_0]], %[[VAL_2]]{{\[}}%[[VAL_3]] : i32] : vector<1xf32>
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : vector<1xf32> to vector<f32>
// CHECK:           return %[[VAL_5]] : vector<f32>
// CHECK:         }
func.func @insertelement_into_vec_0d_f32(%arg0: f32, %arg1: vector<f32>) -> vector<f32> {
  %1 = vector.insertelement %arg0, %arg1[] : vector<f32>
  return %1 : vector<f32>
}

// -----

// CHECK-LABEL:   func.func @type_cast_f32(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : memref<8x8x8xf32> to !llvm.struct<(ptr, ptr, i32, array<3 x i32>, array<3 x i32>)>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i32)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<(ptr, ptr, i32, array<3 x i32>, array<3 x i32>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][0] : !llvm.struct<(ptr, ptr, i32)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.struct<(ptr, ptr, i32, array<3 x i32>, array<3 x i32>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_4]][1] : !llvm.struct<(ptr, ptr, i32)>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_6]][2] : !llvm.struct<(ptr, ptr, i32)>
// CHECK:           %[[VAL_9:.*]] = builtin.unrealized_conversion_cast %[[VAL_8]] : !llvm.struct<(ptr, ptr, i32)> to memref<vector<8x8x8xf32>>
// CHECK:           return %[[VAL_9]] : memref<vector<8x8x8xf32>>
// CHECK:         }
func.func @type_cast_f32(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
  return %0 : memref<vector<8x8x8xf32>>
}

// -----

// CHECK-LABEL:   func.func @type_cast_non_zero_addrspace(
// CHECK-SAME:                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<8x8x8xf32, 3>) -> memref<vector<8x8x8xf32>, 3> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : memref<8x8x8xf32, 3> to !llvm.struct<(ptr<3>, ptr<3>, i32, array<3 x i32>, array<3 x i32>)>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<3>, ptr<3>, i32)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<3 x i32>, array<3 x i32>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][0] : !llvm.struct<(ptr<3>, ptr<3>, i32)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<3 x i32>, array<3 x i32>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_4]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32)>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_6]][2] : !llvm.struct<(ptr<3>, ptr<3>, i32)>
// CHECK:           %[[VAL_9:.*]] = builtin.unrealized_conversion_cast %[[VAL_8]] : !llvm.struct<(ptr<3>, ptr<3>, i32)> to memref<vector<8x8x8xf32>, 3>
// CHECK:           return %[[VAL_9]] : memref<vector<8x8x8xf32>, 3>
// CHECK:         }
func.func @type_cast_non_zero_addrspace(%arg0: memref<8x8x8xf32, 3>) -> memref<vector<8x8x8xf32>, 3> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32, 3> to memref<vector<8x8x8xf32>, 3>
  return %0 : memref<vector<8x8x8xf32>, 3>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vec1d_from_index(
// CHECK-SAME:                                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) -> vector<2xindex> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : index to i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.insertelement %[[VAL_1]], %[[VAL_2]]{{\[}}%[[VAL_3]] : i32] : vector<2xi32>
// CHECK:           %[[VAL_5:.*]] = llvm.shufflevector %[[VAL_4]], %[[VAL_2]] [0, 0] : vector<2xi32>
// CHECK:           %[[VAL_6:.*]] = builtin.unrealized_conversion_cast %[[VAL_5]] : vector<2xi32> to vector<2xindex>
// CHECK:           return %[[VAL_6]] : vector<2xindex>
// CHECK:         }
func.func @broadcast_vec1d_from_index(%arg0: index) -> vector<2xindex> {
  %0 = vector.broadcast %arg0 : index to vector<2xindex>
  return %0 : vector<2xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vec2d_from_vec0d(
// CHECK-SAME:                                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<f32>) -> vector<3x2xf32> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<f32> to vector<1xf32>
// CHECK:           %[[VAL_2:.*]] = ub.poison : vector<3x2xf32>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_4]] : i32] : vector<1xf32>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.poison : vector<2xf32>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.insertelement %[[VAL_5]], %[[VAL_6]]{{\[}}%[[VAL_7]] : i32] : vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = llvm.shufflevector %[[VAL_8]], %[[VAL_6]] [0, 0] : vector<2xf32>
// CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_3]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_10]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_11]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:           %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_12]] : !llvm.array<3 x vector<2xf32>> to vector<3x2xf32>
// CHECK:           return %[[VAL_13]] : vector<3x2xf32>
// CHECK:         }
func.func @broadcast_vec2d_from_vec0d(%arg0: vector<f32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vec2d_from_index_vec1d(
// CHECK-SAME:                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<2xindex>) -> vector<3x2xindex> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<2xindex> to vector<2xi32>
// CHECK:           %[[VAL_2:.*]] = ub.poison : vector<3x2xindex>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : vector<3x2xindex> to !llvm.array<3 x vector<2xi32>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_3]][0] : !llvm.array<3 x vector<2xi32>>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_4]][1] : !llvm.array<3 x vector<2xi32>>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_5]][2] : !llvm.array<3 x vector<2xi32>>
// CHECK:           %[[VAL_7:.*]] = builtin.unrealized_conversion_cast %[[VAL_6]] : !llvm.array<3 x vector<2xi32>> to vector<3x2xindex>
// CHECK:           return %[[VAL_7]] : vector<3x2xindex>
// CHECK:         }
func.func @broadcast_vec2d_from_index_vec1d(%arg0: vector<2xindex>) -> vector<3x2xindex> {
  %0 = vector.broadcast %arg0 : vector<2xindex> to vector<3x2xindex>
  return %0 : vector<3x2xindex>
}

// -----

// CHECK-LABEL:   func.func @outerproduct_index(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<2xindex>,
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<3xindex>) -> vector<2x3xindex> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<2xindex> to vector<2xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0> : vector<2x3xindex>
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_3]] : vector<2x3xindex> to !llvm.array<2 x vector<3xi32>>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_5]] : i64] : vector<2xi32>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.poison : vector<3xi32>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_9:.*]] = llvm.insertelement %[[VAL_6]], %[[VAL_7]]{{\[}}%[[VAL_8]] : i32] : vector<3xi32>
// CHECK:           %[[VAL_10:.*]] = llvm.shufflevector %[[VAL_9]], %[[VAL_7]] [0, 0, 0] : vector<3xi32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_10]] : vector<3xi32> to vector<3xindex>
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_1]] : vector<3xindex>
// CHECK:           %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_12]] : vector<3xindex> to vector<3xi32>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_4]][0] : !llvm.array<2 x vector<3xi32>>
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_16:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_15]] : i64] : vector<2xi32>
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.poison : vector<3xi32>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.insertelement %[[VAL_16]], %[[VAL_17]]{{\[}}%[[VAL_18]] : i32] : vector<3xi32>
// CHECK:           %[[VAL_20:.*]] = llvm.shufflevector %[[VAL_19]], %[[VAL_17]] [0, 0, 0] : vector<3xi32>
// CHECK:           %[[VAL_21:.*]] = builtin.unrealized_conversion_cast %[[VAL_20]] : vector<3xi32> to vector<3xindex>
// CHECK:           %[[VAL_22:.*]] = arith.muli %[[VAL_21]], %[[VAL_1]] : vector<3xindex>
// CHECK:           %[[VAL_23:.*]] = builtin.unrealized_conversion_cast %[[VAL_22]] : vector<3xindex> to vector<3xi32>
// CHECK:           %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_14]][1] : !llvm.array<2 x vector<3xi32>>
// CHECK:           %[[VAL_25:.*]] = builtin.unrealized_conversion_cast %[[VAL_24]] : !llvm.array<2 x vector<3xi32>> to vector<2x3xindex>
// CHECK:           return %[[VAL_25]] : vector<2x3xindex>
// CHECK:         }
func.func @outerproduct_index(%arg0: vector<2xindex>, %arg1: vector<3xindex>) -> vector<2x3xindex> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xindex>, vector<3xindex>
  return %2 : vector<2x3xindex>
}

// -----

// CHECK-LABEL:   func.func @extract_strided_slice_index_1d_from_1d(
// CHECK-SAME:                                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<4xindex>) -> vector<2xindex> {
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<4xindex> to vector<4xi32>
// CHECK:           %[[VAL_2:.*]] = llvm.shufflevector %[[VAL_1]], %[[VAL_1]] [2, 3] : vector<4xi32>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : vector<2xi32> to vector<2xindex>
// CHECK:           return %[[VAL_3]] : vector<2xindex>
// CHECK:         }
func.func @extract_strided_slice_index_1d_from_1d(%arg0: vector<4xindex>) -> vector<2xindex> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xindex> to vector<2xindex>
  return %0 : vector<2xindex>
}

// -----

// CHECK-LABEL:   func.func @insert_strided_index_slice_index_2d_into_3d(
// CHECK-SAME:                                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<4x4xindex>,
// CHECK-SAME:                                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<4x4x4xindex>) -> vector<4x4x4xindex> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<4x4x4xindex> to !llvm.array<4 x array<4 x vector<4xi32>>>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<4x4xindex> to !llvm.array<4 x vector<4xi32>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][2] : !llvm.array<4 x array<4 x vector<4xi32>>>
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : !llvm.array<4 x array<4 x vector<4xi32>>> to vector<4x4x4xindex>
// CHECK:           return %[[VAL_5]] : vector<4x4x4xindex>
// CHECK:         }
func.func @insert_strided_index_slice_index_2d_into_3d(%b: vector<4x4xindex>, %c: vector<4x4x4xindex>) -> vector<4x4x4xindex> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x4xindex> into vector<4x4x4xindex>
  return %0 : vector<4x4x4xindex>
}

// -----

// CHECK-LABEL:   func.func @matrix_ops_index(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<64xindex>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<48xindex>) -> vector<12xindex> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<48xindex> to vector<48xi32>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : vector<64xindex> to vector<64xi32>
// CHECK:           %[[VAL_4:.*]] = llvm.intr.matrix.multiply %[[VAL_3]], %[[VAL_2]] {lhs_columns = 16 : i32, lhs_rows = 4 : i32, rhs_columns = 3 : i32} : (vector<64xi32>, vector<48xi32>) -> vector<12xi32>
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : vector<12xi32> to vector<12xindex>
// CHECK:           return %[[VAL_5]] : vector<12xindex>
// CHECK:         }
func.func @matrix_ops_index(%A: vector<64xindex>, %B: vector<48xindex>) -> vector<12xindex> {
  %C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
    (vector<64xindex>, vector<48xindex>) -> vector<12xindex>
  return %C: vector<12xindex>
}

// -----

// CHECK-LABEL:   func.func @transfer_read_write_index_1d(
// CHECK-SAME:                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xindex>,
// CHECK-SAME:                                            %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) -> vector<17xindex> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : index to i32
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : memref<?xindex> to !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<7> : vector<17xindex>
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : vector<17xindex> to vector<17xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : vector<17xi32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_0]], %[[VAL_7]] : memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_1]] : index
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : index to i32
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.poison : vector<17xi32>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.insertelement %[[VAL_10]], %[[VAL_11]]{{\[}}%[[VAL_12]] : i32] : vector<17xi32>
// CHECK:           %[[VAL_14:.*]] = llvm.shufflevector %[[VAL_13]], %[[VAL_11]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<17xi32>
// CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_6]] : vector<17xi32>
// CHECK:           %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_3]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_16]]{{\[}}%[[VAL_2]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:           %[[VAL_18:.*]] = llvm.intr.masked.load %[[VAL_17]], %[[VAL_15]], %[[VAL_5]] {alignment = 4 : i32} : (!llvm.ptr, vector<17xi1>, vector<17xi32>) -> vector<17xi32>
// CHECK:           %[[VAL_19:.*]] = builtin.unrealized_conversion_cast %[[VAL_18]] : vector<17xi32> to vector<17xindex>
// CHECK:           %[[VAL_20:.*]] = memref.dim %[[VAL_0]], %[[VAL_7]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_1]] : index
// CHECK:           %[[VAL_22:.*]] = arith.index_cast %[[VAL_21]] : index to i32
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.poison : vector<17xi32>
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.insertelement %[[VAL_22]], %[[VAL_23]]{{\[}}%[[VAL_24]] : i32] : vector<17xi32>
// CHECK:           %[[VAL_26:.*]] = llvm.shufflevector %[[VAL_25]], %[[VAL_23]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<17xi32>
// CHECK:           %[[VAL_27:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_6]] : vector<17xi32>
// CHECK:           %[[VAL_28:.*]] = llvm.extractvalue %[[VAL_3]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_28]]{{\[}}%[[VAL_2]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:           llvm.intr.masked.store %[[VAL_18]], %[[VAL_29]], %[[VAL_27]] {alignment = 4 : i32} : vector<17xi32>, vector<17xi1> into !llvm.ptr
// CHECK:           return %[[VAL_19]] : vector<17xindex>
// CHECK:         }
func.func @transfer_read_write_index_1d(%A : memref<?xindex>, %base: index) -> vector<17xindex> {
  %f7 = arith.constant 7: index
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xindex>, vector<17xindex>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xindex>, memref<?xindex>
  return %f: vector<17xindex>
}

// -----

// CHECK-LABEL:   func.func @transfer_read_write_1d_non_zero_addrspace(
// CHECK-SAME:                                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32, 3>,
// CHECK-SAME:                                                         %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) -> vector<17xf32> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : index to i32
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : memref<?xf32, 3> to !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<7.000000e+00> : vector<17xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : vector<17xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_0]], %[[VAL_6]] : memref<?xf32, 3>
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_1]] : index
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : index to i32
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.poison : vector<17xi32>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.insertelement %[[VAL_9]], %[[VAL_10]]{{\[}}%[[VAL_11]] : i32] : vector<17xi32>
// CHECK:           %[[VAL_13:.*]] = llvm.shufflevector %[[VAL_12]], %[[VAL_10]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<17xi32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_5]] : vector<17xi32>
// CHECK:           %[[VAL_15:.*]] = llvm.extractvalue %[[VAL_3]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_15]]{{\[}}%[[VAL_2]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_17:.*]] = llvm.intr.masked.load %[[VAL_16]], %[[VAL_14]], %[[VAL_4]] {alignment = 4 : i32} : (!llvm.ptr<3>, vector<17xi1>, vector<17xf32>) -> vector<17xf32>
// CHECK:           %[[VAL_18:.*]] = memref.dim %[[VAL_0]], %[[VAL_6]] : memref<?xf32, 3>
// CHECK:           %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_1]] : index
// CHECK:           %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : index to i32
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.poison : vector<17xi32>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_23:.*]] = llvm.insertelement %[[VAL_20]], %[[VAL_21]]{{\[}}%[[VAL_22]] : i32] : vector<17xi32>
// CHECK:           %[[VAL_24:.*]] = llvm.shufflevector %[[VAL_23]], %[[VAL_21]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<17xi32>
// CHECK:           %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_5]] : vector<17xi32>
// CHECK:           %[[VAL_26:.*]] = llvm.extractvalue %[[VAL_3]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_26]]{{\[}}%[[VAL_2]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
// CHECK:           llvm.intr.masked.store %[[VAL_17]], %[[VAL_27]], %[[VAL_25]] {alignment = 4 : i32} : vector<17xf32>, vector<17xi1> into !llvm.ptr<3>
// CHECK:           return %[[VAL_17]] : vector<17xf32>
// CHECK:         }
func.func @transfer_read_write_1d_non_zero_addrspace(%A : memref<?xf32, 3>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32, 3>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32, 3>
  return %f: vector<17xf32>
}

// -----

// CHECK-LABEL:   func.func @transfer_read_1d_inbounds(
// CHECK-SAME:                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                                         %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) -> vector<17xf32> {
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : index to i32
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : memref<?xf32> to !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_3]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_2]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] {alignment = 4 : i64} : !llvm.ptr -> vector<17xf32>
// CHECK:           return %[[VAL_6]] : vector<17xf32>
// CHECK:         }
func.func @transfer_read_1d_inbounds(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7 {in_bounds = [true]} :
    memref<?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
