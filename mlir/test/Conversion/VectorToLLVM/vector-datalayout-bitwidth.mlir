// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec< #dlti.dl_entry<index, 32>>} {
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
}
