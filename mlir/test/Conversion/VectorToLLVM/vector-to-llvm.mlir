// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s

//===========================================================================//
// Complex tests for Vector-to-LLVM conversion
//
// These examples, in order to convert to LLVM, require
//  * `populateVectorToLLVMConversionPatterns`.
// as well as various other patterns/conversion that are part of
// `ConvertVectorToLLVMPass`.
//
// Please, in the first instance, always try adding tests in
// vector-to-llvm-interface.mlir instead.
//===========================================================================//

//===----------------------------------------------------------------------===//
// vector.bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func.func @bitcast_2d(
// CHECK-SAME:      %[[ARG_0:.*]]: vector<2x4xi32>) -> vector<2x2xi64> {
// CHECK:           %[[T0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : vector<2x4xi32> to !llvm.array<2 x vector<4xi32>>
// CHECK:           %[[VEC_1:.*]] = llvm.extractvalue %[[T0]][0] : !llvm.array<2 x vector<4xi32>>
// CHECK:           %[[BCAST_1:.*]] = llvm.bitcast %[[VEC_1]] : vector<4xi32> to vector<2xi64>
// CHECK:           %[[OUT_1:.*]] = llvm.insertvalue %[[BCAST_1]], {{.*}}[0] : !llvm.array<2 x vector<2xi64>>
// CHECK:           %[[VEC_2:.*]] = llvm.extractvalue %[[T0]][1] : !llvm.array<2 x vector<4xi32>>
// CHECK:           %[[BCAST_2:.*]] = llvm.bitcast %[[VEC_2]] : vector<4xi32> to vector<2xi64>
// CHECK:           %[[OUT_2:.*]] = llvm.insertvalue %[[BCAST_2]], %[[OUT_1]][1] : !llvm.array<2 x vector<2xi64>>
func.func @bitcast_2d(%arg0: vector<2x4xi32>) -> vector<2x2xi64> {
  %0 = vector.bitcast %arg0 : vector<2x4xi32> to vector<2x2xi64>
  return %0 : vector<2x2xi64>
}

// -----

// CHECK-LABEL:   func.func @bitcast_2d_scalable(
// CHECK-SAME:      %[[ARG_0:.*]]: vector<2x[4]xi32>) -> vector<2x[2]xi64> {
// CHECK:           %[[T0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : vector<2x[4]xi32> to !llvm.array<2 x vector<[4]xi32>>
// CHECK:           %[[VEC_1:.*]] = llvm.extractvalue %[[T0]][0] : !llvm.array<2 x vector<[4]xi32>>
// CHECK:           %[[BCAST_1:.*]] = llvm.bitcast %[[VEC_1]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK:           %[[OUT_1:.*]] = llvm.insertvalue %[[BCAST_1]], {{.*}}[0] : !llvm.array<2 x vector<[2]xi64>>
// CHECK:           %[[VEC_2:.*]] = llvm.extractvalue %[[T0]][1] : !llvm.array<2 x vector<[4]xi32>>
// CHECK:           %[[BCAST_2:.*]] = llvm.bitcast %[[VEC_2]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK:           %[[OUT_2:.*]] = llvm.insertvalue %[[BCAST_2]], %[[OUT_1]][1] : !llvm.array<2 x vector<[2]xi64>>
func.func @bitcast_2d_scalable(%arg0: vector<2x[4]xi32>) -> vector<2x[2]xi64> {
  %0 = vector.bitcast %arg0 : vector<2x[4]xi32> to vector<2x[2]xi64>
  return %0 : vector<2x[2]xi64>
}

// -----

//===----------------------------------------------------------------------===//
// vector.broadcast
//===----------------------------------------------------------------------===//

func.func @broadcast_vec0d_from_f32(%arg0: f32) -> vector<f32> {
  %0 = vector.broadcast %arg0 : f32 to vector<f32>
  return %0 : vector<f32>
}
// CHECK-LABEL: @broadcast_vec0d_from_f32
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<1xf32> to vector<f32>
// CHECK:       return %[[T1]] : vector<f32>

// -----

func.func @broadcast_vec1d_from_f32(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: @broadcast_vec1d_from_f32
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       return %[[T1]] : vector<2xf32>

// -----

func.func @broadcast_single_elem_vec1d_from_f32(%arg0: f32) -> vector<1xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<1xf32>
  return %0 : vector<1xf32>
}
// CHECK-LABEL: @broadcast_single_elem_vec1d_from_f32
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK-NOT:   llvm.shufflevector
// CHECK:       return %[[T0]] : vector<1xf32>

// -----

func.func @broadcast_vec1d_from_f32_scalable(%arg0: f32) -> vector<[2]xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<[2]xf32>
  return %0 : vector<[2]xf32>
}
// CHECK-LABEL: @broadcast_vec1d_from_f32_scalable
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       return %[[T1]] : vector<[2]xf32>

// -----

func.func @broadcast_vec1d_from_index(%arg0: index) -> vector<2xindex> {
  %0 = vector.broadcast %arg0 : index to vector<2xindex>
  return %0 : vector<2xindex>
}
// CHECK-LABEL: @broadcast_vec1d_from_index
// CHECK-SAME:  %[[A:.*]]: index)
// CHECK:       %[[A1:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A1]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<2xi64> to vector<2xindex>
// CHECK:       return %[[T2]] : vector<2xindex>

// -----

func.func @broadcast_vec1d_from_index_scalable(%arg0: index) -> vector<[2]xindex> {
  %0 = vector.broadcast %arg0 : index to vector<[2]xindex>
  return %0 : vector<[2]xindex>
}
// CHECK-LABEL: @broadcast_vec1d_from_index_scalable
// CHECK-SAME:  %[[A:.*]]: index)
// CHECK:       %[[A1:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A1]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<[2]xi64> to vector<[2]xindex>
// CHECK:       return %[[T2]] : vector<[2]xindex>

// -----

func.func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : !llvm.array<2 x vector<3xf32>> to vector<2x3xf32>
// CHECK:       return %[[T4]] : vector<2x3xf32>

// -----

func.func @broadcast_vec2d_from_scalar_scalable(%arg0: f32) -> vector<2x[3]xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_scalar_scalable(
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[0] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[1] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : !llvm.array<2 x vector<[3]xf32>> to vector<2x[3]xf32>
// CHECK:       return %[[T4]] : vector<2x[3]xf32>

// -----

func.func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[0, 0] : !llvm.array<2 x array<3 x vector<4xf32>>>
// ...
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[1, 2] : !llvm.array<2 x array<3 x vector<4xf32>>>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : !llvm.array<2 x array<3 x vector<4xf32>>> to vector<2x3x4xf32>
// CHECK:       return %[[T4]] : vector<2x3x4xf32>

// -----

func.func @broadcast_vec3d_from_scalar_scalable(%arg0: f32) -> vector<2x3x[4]xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x[4]xf32>
  return %0 : vector<2x3x[4]xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_scalar_scalable(
// CHECK-SAME:  %[[A:.*]]: f32)
// CHECK:       %[[T0:.*]] = llvm.insertelement %[[A]]
// CHECK:       %[[T1:.*]] = llvm.shufflevector %[[T0]]
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[0, 0] : !llvm.array<2 x array<3 x vector<[4]xf32>>>
// ...
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %{{.*}}[1, 2] : !llvm.array<2 x array<3 x vector<[4]xf32>>>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : !llvm.array<2 x array<3 x vector<[4]xf32>>> to vector<2x3x[4]xf32>
// CHECK:       return %[[T4]] : vector<2x3x[4]xf32>

// -----

func.func @broadcast_vec2d_from_vec0d(%arg0: vector<f32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_vec0d(
// CHECK-SAME:  %[[A:.*]]: vector<f32>)
//       CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<f32> to vector<1xf32>
//       CHECK: %[[T1:.*]] = ub.poison : vector<3x2xf32>
//       CHECK: %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T4:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK: %[[T5:.*]] = llvm.extractelement %[[T0]][%[[T4]] : i64] : vector<1xf32>
//       CHECK: %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
//       CHECK: %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
//       CHECK: %[[T7:.*]] = llvm.insertvalue %[[T6]], %[[T2]][0] : !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][1] : !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T9:.*]] = llvm.insertvalue %[[T6]], %[[T8]][2] : !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T9]] : !llvm.array<3 x vector<2xf32>> to vector<3x2xf32>
//       CHECK: return %[[T10]] : vector<3x2xf32>

// -----

func.func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>)
// CHECK:       %[[T0:.*]] = ub.poison : vector<3x2xf32>
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[A]], %[[T1]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : !llvm.array<3 x vector<2xf32>> to vector<3x2xf32>
// CHECK:       return %[[T5]] : vector<3x2xf32>

// -----

func.func @broadcast_vec2d_from_vec1d_scalable(%arg0: vector<[2]xf32>) -> vector<3x[2]xf32> {
  %0 = vector.broadcast %arg0 : vector<[2]xf32> to vector<3x[2]xf32>
  return %0 : vector<3x[2]xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_vec1d_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<[2]xf32>)
// CHECK:       %[[T0:.*]] = ub.poison : vector<3x[2]xf32>
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x[2]xf32> to !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[A]], %[[T1]][0] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][1] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][2] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : !llvm.array<3 x vector<[2]xf32>> to vector<3x[2]xf32>
// CHECK:       return %[[T5]] : vector<3x[2]xf32>

// -----

func.func @broadcast_vec2d_from_index_vec1d(%arg0: vector<2xindex>) -> vector<3x2xindex> {
  %0 = vector.broadcast %arg0 : vector<2xindex> to vector<3x2xindex>
  return %0 : vector<3x2xindex>
}
// CHECK-LABEL: @broadcast_vec2d_from_index_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xindex>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2xindex> to vector<2xi64>
// CHECK:       %[[T0:.*]] = ub.poison : vector<3x2xindex>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xindex> to !llvm.array<3 x vector<2xi64>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][0] : !llvm.array<3 x vector<2xi64>>

// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.array<3 x vector<2xi64>> to vector<3x2xindex>
// CHECK:       return %[[T4]] : vector<3x2xindex>

// -----

func.func @broadcast_vec2d_from_index_vec1d_scalable(%arg0: vector<[2]xindex>) -> vector<3x[2]xindex> {
  %0 = vector.broadcast %arg0 : vector<[2]xindex> to vector<3x[2]xindex>
  return %0 : vector<3x[2]xindex>
}
// CHECK-LABEL: @broadcast_vec2d_from_index_vec1d_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<[2]xindex>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<[2]xindex> to vector<[2]xi64>
// CHECK:       %[[T0:.*]] = ub.poison : vector<3x[2]xindex>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x[2]xindex> to !llvm.array<3 x vector<[2]xi64>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][0] : !llvm.array<3 x vector<[2]xi64>>

// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.array<3 x vector<[2]xi64>> to vector<3x[2]xindex>
// CHECK:       return %[[T4]] : vector<3x[2]xindex>

// -----

func.func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>)
// CHECK-DAG:   %[[T0:.*]] = ub.poison : vector<3x2xf32>
// CHECK-DAG:   %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK-DAG:   %[[T1:.*]] = ub.poison : vector<4x3x2xf32>
// CHECK-DAG:   %[[T6:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>

// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[A]], %[[T4]][2] : !llvm.array<3 x vector<2xf32>>

// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T5]], %[[T6]][0] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T5]], %[[T7]][1] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T5]], %[[T8]][2] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T5]], %[[T9]][3] : !llvm.array<4 x array<3 x vector<2xf32>>>

// CHECK:       %[[T11:.*]] = builtin.unrealized_conversion_cast %[[T10]] : !llvm.array<4 x array<3 x vector<2xf32>>> to vector<4x3x2xf32>
// CHECK:       return %[[T11]] : vector<4x3x2xf32>

// -----

func.func @broadcast_vec3d_from_vec1d_scalable(%arg0: vector<[2]xf32>) -> vector<4x3x[2]xf32> {
  %0 = vector.broadcast %arg0 : vector<[2]xf32> to vector<4x3x[2]xf32>
  return %0 : vector<4x3x[2]xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_vec1d_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<[2]xf32>)
// CHECK-DAG:   %[[T0:.*]] = ub.poison : vector<3x[2]xf32>
// CHECK-DAG:   %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x[2]xf32> to !llvm.array<3 x vector<[2]xf32>>
// CHECK-DAG:   %[[T1:.*]] = ub.poison : vector<4x3x[2]xf32>
// CHECK-DAG:   %[[T6:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x[2]xf32> to !llvm.array<4 x array<3 x vector<[2]xf32>>>

// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][0] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][1] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[A]], %[[T4]][2] : !llvm.array<3 x vector<[2]xf32>>

// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T5]], %[[T6]][0] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T5]], %[[T7]][1] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T5]], %[[T8]][2] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T5]], %[[T9]][3] : !llvm.array<4 x array<3 x vector<[2]xf32>>>

// CHECK:       %[[T11:.*]] = builtin.unrealized_conversion_cast %[[T10]] : !llvm.array<4 x array<3 x vector<[2]xf32>>> to vector<4x3x[2]xf32>
// CHECK:       return %[[T11]] : vector<4x3x[2]xf32>

// -----

func.func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_vec2d(
// CHECK-SAME:  %[[A:.*]]: vector<3x2xf32>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T0:.*]] = ub.poison : vector<4x3x2xf32>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][0] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T1]], %[[T3]][1] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T1]], %[[T5]][2] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T1]], %[[T7]][3] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T9]] : !llvm.array<4 x array<3 x vector<2xf32>>> to vector<4x3x2xf32>
// CHECK:       return %[[T10]] : vector<4x3x2xf32>

// -----

func.func @broadcast_vec3d_from_vec2d_scalable(%arg0: vector<3x[2]xf32>) -> vector<4x3x[2]xf32> {
  %0 = vector.broadcast %arg0 : vector<3x[2]xf32> to vector<4x3x[2]xf32>
  return %0 : vector<4x3x[2]xf32>
}
// CHECK-LABEL: @broadcast_vec3d_from_vec2d_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<3x[2]xf32>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<3x[2]xf32> to !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T0:.*]] = ub.poison : vector<4x3x[2]xf32>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<4x3x[2]xf32> to !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][0] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T1]], %[[T3]][1] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T1]], %[[T5]][2] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T1]], %[[T7]][3] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T9]] : !llvm.array<4 x array<3 x vector<[2]xf32>>> to vector<4x3x[2]xf32>
// CHECK:       return %[[T10]] : vector<4x3x[2]xf32>


// -----

func.func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: @broadcast_stretch(
// CHECK-SAME:  %[[A:.*]]: vector<1xf32>)
// CHECK:       %[[T1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T2:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T1]] : i64] : vector<1xf32>
// CHECK:       %[[T3:.*]] = llvm.insertelement %[[T2]]
// CHECK:       %[[T4:.*]] = llvm.shufflevector %[[T3]]
// CHECK:       return %[[T4]] : vector<4xf32>

// -----

func.func @broadcast_stretch_scalable(%arg0: vector<1xf32>) -> vector<[4]xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<[4]xf32>
  return %0 : vector<[4]xf32>
}
// CHECK-LABEL: @broadcast_stretch_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<1xf32>)
// CHECK:       %[[T1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T2:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T1]] : i64] : vector<1xf32>
// CHECK:       %[[T3:.*]] = llvm.insertelement %[[T2]]
// CHECK:       %[[T4:.*]] = llvm.shufflevector %[[T3]]
// CHECK:       return %[[T4]] : vector<[4]xf32>

// -----

func.func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: @broadcast_stretch_at_start(
// CHECK-SAME:  %[[A:.*]]: vector<1x4xf32>)
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<1x4xf32> to !llvm.array<1 x vector<4xf32>>
// CHECK:       %[[T1:.*]] = ub.poison : vector<3x4xf32>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<3x4xf32> to !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T3:.*]] = llvm.extractvalue %[[T2]][0] : !llvm.array<1 x vector<4xf32>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T3]], %[[T4]][0] : !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T3]], %[[T5]][1] : !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T3]], %[[T6]][2] : !llvm.array<3 x vector<4xf32>>
// CHECK:       %[[T8:.*]] = builtin.unrealized_conversion_cast %[[T7]] : !llvm.array<3 x vector<4xf32>> to vector<3x4xf32>
// CHECK:       return %[[T8]] : vector<3x4xf32>

// -----

func.func @broadcast_stretch_at_start_scalable(%arg0: vector<1x[4]xf32>) -> vector<3x[4]xf32> {
  %0 = vector.broadcast %arg0 : vector<1x[4]xf32> to vector<3x[4]xf32>
  return %0 : vector<3x[4]xf32>
}
// CHECK-LABEL: @broadcast_stretch_at_start_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<1x[4]xf32>)
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<1x[4]xf32> to !llvm.array<1 x vector<[4]xf32>>
// CHECK:       %[[T1:.*]] = ub.poison : vector<3x[4]xf32>
// CHECK:       %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<3x[4]xf32> to !llvm.array<3 x vector<[4]xf32>>
// CHECK:       %[[T3:.*]] = llvm.extractvalue %[[T2]][0] : !llvm.array<1 x vector<[4]xf32>>
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T3]], %[[T4]][0] : !llvm.array<3 x vector<[4]xf32>>
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T3]], %[[T5]][1] : !llvm.array<3 x vector<[4]xf32>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T3]], %[[T6]][2] : !llvm.array<3 x vector<[4]xf32>>
// CHECK:       %[[T8:.*]] = builtin.unrealized_conversion_cast %[[T7]] : !llvm.array<3 x vector<[4]xf32>> to vector<3x[4]xf32>
// CHECK:       return %[[T8]] : vector<3x[4]xf32>

// -----

func.func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}
// CHECK-LABEL: @broadcast_stretch_at_end(
// CHECK-SAME:  %[[A:.*]]: vector<4x1xf32>)
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4x1xf32> to !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T1:.*]] = ub.poison : vector<4x3xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3xf32> to !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = llvm.extractvalue %[[T2]][0] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T5:.*]] = llvm.extractelement %[[T3]]{{\[}}%[[T4]] : i64] : vector<1xf32>
// CHECK:       %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
// CHECK:       %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][0] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T10:.*]] = llvm.extractvalue %[[T2]][1] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T11:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T12:.*]] = llvm.extractelement %[[T10]]{{\[}}%[[T11]] : i64] : vector<1xf32>
// CHECK:       %[[T13Insert:.*]] = llvm.insertelement %[[T12]]
// CHECK:       %[[T13:.*]] = llvm.shufflevector %[[T13Insert]]
// CHECK:       %[[T14:.*]] = llvm.insertvalue %[[T13]], %[[T8]][1] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T16:.*]] = llvm.extractvalue %[[T2]][2] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T17:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T18:.*]] = llvm.extractelement %[[T16]]{{\[}}%[[T17]] : i64] : vector<1xf32>
// CHECK:       %[[T19Insert:.*]] = llvm.insertelement %[[T18]]
// CHECK:       %[[T19:.*]] = llvm.shufflevector %[[T19Insert]]
// CHECK:       %[[T20:.*]] = llvm.insertvalue %[[T19]], %[[T14]][2] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T22:.*]] = llvm.extractvalue %[[T2]][3] : !llvm.array<4 x vector<1xf32>>
// CHECK:       %[[T23:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T24:.*]] = llvm.extractelement %[[T22]]{{\[}}%[[T23]] : i64] : vector<1xf32>
// CHECK:       %[[T25Insert:.*]] = llvm.insertelement %[[T24]]
// CHECK:       %[[T25:.*]] = llvm.shufflevector %[[T25Insert]]
// CHECK:       %[[T26:.*]] = llvm.insertvalue %[[T25]], %[[T20]][3] : !llvm.array<4 x vector<3xf32>>
// CHECK:       %[[T27:.*]] = builtin.unrealized_conversion_cast %[[T26]] : !llvm.array<4 x vector<3xf32>> to vector<4x3xf32>
// CHECK:       return %[[T27]] : vector<4x3xf32>

// TODO: Add support for scalable vectors

func.func @broadcast_stretch_at_end_scalable(%arg0: vector<[4]x1xf32>) -> vector<[4]x3xf32> {
  %0 = vector.broadcast %arg0 : vector<[4]x1xf32> to vector<[4]x3xf32>
  return %0 : vector<[4]x3xf32>
}
// CHECK-LABEL: @broadcast_stretch_at_end_scalable
// CHECK-SAME:  %[[A:.*]]: vector<[4]x1xf32>)
// CHECK: vector.broadcast %[[A]] : vector<[4]x1xf32> to vector<[4]x3xf32>

// -----

func.func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: @broadcast_stretch_in_middle(
// CHECK-SAME:  %[[A:.*]]: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
// CHECK:       %[[T3:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4x1x2xf32> to !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T1:.*]] = ub.poison : vector<4x3x2xf32>
// CHECK:       %[[T9:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T2:.*]] = ub.poison : vector<3x2xf32>
// CHECK:       %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T4:.*]] = llvm.extractvalue %[[T3]][0, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T4]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T4]], %[[T6]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T4]], %[[T7]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T8]], %[[T9]][0] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T12:.*]] = llvm.extractvalue %[[T3]][1, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T14:.*]] = llvm.insertvalue %[[T12]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T15:.*]] = llvm.insertvalue %[[T12]], %[[T14]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T16:.*]] = llvm.insertvalue %[[T12]], %[[T15]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T17:.*]] = llvm.insertvalue %[[T16]], %[[T10]][1] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T19:.*]] = llvm.extractvalue %[[T3]][2, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T21:.*]] = llvm.insertvalue %[[T19]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T22:.*]] = llvm.insertvalue %[[T19]], %[[T21]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T23:.*]] = llvm.insertvalue %[[T19]], %[[T22]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T24:.*]] = llvm.insertvalue %[[T23]], %[[T17]][2] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T26:.*]] = llvm.extractvalue %[[T3]][3, 0] : !llvm.array<4 x array<1 x vector<2xf32>>>
// CHECK:       %[[T28:.*]] = llvm.insertvalue %[[T26]], %[[T5]][0] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T29:.*]] = llvm.insertvalue %[[T26]], %[[T28]][1] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T30:.*]] = llvm.insertvalue %[[T26]], %[[T29]][2] : !llvm.array<3 x vector<2xf32>>
// CHECK:       %[[T31:.*]] = llvm.insertvalue %[[T30]], %[[T24]][3] : !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T32:.*]] = builtin.unrealized_conversion_cast %[[T31]] : !llvm.array<4 x array<3 x vector<2xf32>>> to vector<4x3x2xf32>
// CHECK:       return %[[T32]] : vector<4x3x2xf32>

// -----

func.func @broadcast_stretch_in_middle_scalable_v1(%arg0: vector<4x1x[2]xf32>) -> vector<4x3x[2]xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x[2]xf32> to vector<4x3x[2]xf32>
  return %0 : vector<4x3x[2]xf32>
}
// CHECK-LABEL: @broadcast_stretch_in_middle_scalable_v1(
// CHECK-SAME:  %[[A:.*]]: vector<4x1x[2]xf32>) -> vector<4x3x[2]xf32> {
// CHECK:       %[[T3:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4x1x[2]xf32> to !llvm.array<4 x array<1 x vector<[2]xf32>>>
// CHECK:       %[[T1:.*]] = ub.poison : vector<4x3x[2]xf32>
// CHECK:       %[[T9:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x[2]xf32> to !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T2:.*]] = ub.poison : vector<3x[2]xf32>
// CHECK:       %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<3x[2]xf32> to !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T4:.*]] = llvm.extractvalue %[[T3]][0, 0] : !llvm.array<4 x array<1 x vector<[2]xf32>>>
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T4]], %[[T5]][0] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T4]], %[[T6]][1] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T4]], %[[T7]][2] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T8]], %[[T9]][0] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T12:.*]] = llvm.extractvalue %[[T3]][1, 0] : !llvm.array<4 x array<1 x vector<[2]xf32>>>
// CHECK:       %[[T14:.*]] = llvm.insertvalue %[[T12]], %[[T5]][0] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T15:.*]] = llvm.insertvalue %[[T12]], %[[T14]][1] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T16:.*]] = llvm.insertvalue %[[T12]], %[[T15]][2] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T17:.*]] = llvm.insertvalue %[[T16]], %[[T10]][1] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T19:.*]] = llvm.extractvalue %[[T3]][2, 0] : !llvm.array<4 x array<1 x vector<[2]xf32>>>
// CHECK:       %[[T21:.*]] = llvm.insertvalue %[[T19]], %[[T5]][0] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T22:.*]] = llvm.insertvalue %[[T19]], %[[T21]][1] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T23:.*]] = llvm.insertvalue %[[T19]], %[[T22]][2] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T24:.*]] = llvm.insertvalue %[[T23]], %[[T17]][2] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T26:.*]] = llvm.extractvalue %[[T3]][3, 0] : !llvm.array<4 x array<1 x vector<[2]xf32>>>
// CHECK:       %[[T28:.*]] = llvm.insertvalue %[[T26]], %[[T5]][0] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T29:.*]] = llvm.insertvalue %[[T26]], %[[T28]][1] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T30:.*]] = llvm.insertvalue %[[T26]], %[[T29]][2] : !llvm.array<3 x vector<[2]xf32>>
// CHECK:       %[[T31:.*]] = llvm.insertvalue %[[T30]], %[[T24]][3] : !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T32:.*]] = builtin.unrealized_conversion_cast %[[T31]] : !llvm.array<4 x array<3 x vector<[2]xf32>>> to vector<4x3x[2]xf32>
// CHECK:       return %[[T32]] : vector<4x3x[2]xf32>

// -----

// TODO: Add support for scalable vectors

func.func @broadcast_stretch_in_middle_scalable_v2(%arg0: vector<[4]x1x2xf32>) -> vector<[4]x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<[4]x1x2xf32> to vector<[4]x3x2xf32>
  return %0 : vector<[4]x3x2xf32>
}
// CHECK-LABEL: @broadcast_stretch_in_middle_scalable_v2(
// CHECK-SAME:  %[[A:.*]]: vector<[4]x1x2xf32>) -> vector<[4]x3x2xf32> {
// CHECK:  vector.broadcast %[[A]] : vector<[4]x1x2xf32> to vector<[4]x3x2xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.outerproduct
//===----------------------------------------------------------------------===//

func.func @outerproduct(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: @outerproduct(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>,
// CHECK-SAME:  %[[B:.*]]: vector<3xf32>)
// CHECK:       %[[T2:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T4:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T3]] : i64] : vector<2xf32>
// CHECK:       %[[T5Insert:.*]] = llvm.insertelement %[[T4]]
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T5Insert]]
// CHECK:       %[[T6:.*]] = arith.mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[T10:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T9]] : i64] : vector<2xf32>
// CHECK:       %[[T11Insert:.*]] = llvm.insertelement %[[T10]]
// CHECK:       %[[T11:.*]] = llvm.shufflevector %[[T11Insert]]
// CHECK:       %[[T12:.*]] = arith.mulf %[[T11]], %[[B]] : vector<3xf32>
// CHECK:       %[[T13:.*]] = llvm.insertvalue %[[T12]], %[[T8]][1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T14:.*]] = builtin.unrealized_conversion_cast %[[T13]] : !llvm.array<2 x vector<3xf32>> to vector<2x3xf32>
// CHECK:       return %[[T14]] : vector<2x3xf32>

// -----

func.func @outerproduct_scalable(%arg0: vector<2xf32>, %arg1: vector<[3]xf32>) -> vector<2x[3]xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<[3]xf32>
  return %2 : vector<2x[3]xf32>
}
// CHECK-LABEL: @outerproduct_scalable
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>,
// CHECK-SAME:  %[[B:.*]]: vector<[3]xf32>)
// CHECK:       %[[T2:.*]] = arith.constant dense<0.000000e+00> : vector<2x[3]xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<2x[3]xf32> to !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T4:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T3]] : i64] : vector<2xf32>
// CHECK:       %[[T5Insert:.*]] = llvm.insertelement %[[T4]]
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T5Insert]]
// CHECK:       %[[T6:.*]] = arith.mulf %[[T5]], %[[B]] : vector<[3]xf32>
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T6]], %[[T7]][0] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[T10:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T9]] : i64] : vector<2xf32>
// CHECK:       %[[T11Insert:.*]] = llvm.insertelement %[[T10]]
// CHECK:       %[[T11:.*]] = llvm.shufflevector %[[T11Insert]]
// CHECK:       %[[T12:.*]] = arith.mulf %[[T11]], %[[B]] : vector<[3]xf32>
// CHECK:       %[[T13:.*]] = llvm.insertvalue %[[T12]], %[[T8]][1] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T14:.*]] = builtin.unrealized_conversion_cast %[[T13]] : !llvm.array<2 x vector<[3]xf32>> to vector<2x[3]xf32>
// CHECK:       return %[[T14]] : vector<2x[3]xf32>

// -----

func.func @outerproduct_index(%arg0: vector<2xindex>, %arg1: vector<3xindex>) -> vector<2x3xindex> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xindex>, vector<3xindex>
  return %2 : vector<2x3xindex>
}
// CHECK-LABEL: @outerproduct_index(
// CHECK-SAME:  %[[A:.*]]: vector<2xindex>,
// CHECK-SAME:  %[[B:.*]]: vector<3xindex>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2xindex> to vector<2xi64>
// CHECK:       %[[T0:.*]] = arith.constant dense<0> : vector<2x3xindex>
// CHECK:       %[[T8:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<2x3xindex> to !llvm.array<2 x vector<3xi64>>
// CHECK:       %[[T2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T3:.*]] = llvm.extractelement %[[T1]]{{\[}}%[[T2]] : i64] : vector<2xi64>
// CHECK:       %[[T4:.*]] = llvm.insertelement %[[T3]]
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T4]]
// CHECK:       %[[T5Cast:.*]] = builtin.unrealized_conversion_cast %[[T5]] : vector<3xi64> to vector<3xindex>
// CHECK:       %[[T6:.*]] = arith.muli %[[T5Cast]], %[[B]] : vector<3xindex>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T6]] : vector<3xindex> to vector<3xi64>
// CHECK:       %{{.*}} = llvm.insertvalue %[[T7]], %[[T8]][0] : !llvm.array<2 x vector<3xi64>>

// -----

func.func @outerproduct_index_scalable(%arg0: vector<2xindex>, %arg1: vector<[3]xindex>) -> vector<2x[3]xindex> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xindex>, vector<[3]xindex>
  return %2 : vector<2x[3]xindex>
}
// CHECK-LABEL: @outerproduct_index_scalable
// CHECK-SAME:  %[[A:.*]]: vector<2xindex>,
// CHECK-SAME:  %[[B:.*]]: vector<[3]xindex>)
// CHECK:       %[[T1:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<2xindex> to vector<2xi64>
// CHECK:       %[[T0:.*]] = arith.constant dense<0> : vector<2x[3]xindex>
// CHECK:       %[[T8:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<2x[3]xindex> to !llvm.array<2 x vector<[3]xi64>>
// CHECK:       %[[T2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T3:.*]] = llvm.extractelement %[[T1]]{{\[}}%[[T2]] : i64] : vector<2xi64>
// CHECK:       %[[T4:.*]] = llvm.insertelement %[[T3]]
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T4]]
// CHECK:       %[[T5Cast:.*]] = builtin.unrealized_conversion_cast %[[T5]] : vector<[3]xi64> to vector<[3]xindex>
// CHECK:       %[[T6:.*]] = arith.muli %[[T5Cast]], %[[B]] : vector<[3]xindex>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[T6]] : vector<[3]xindex> to vector<[3]xi64>
// CHECK:       %{{.*}} = llvm.insertvalue %[[T7]], %[[T8]][0] : !llvm.array<2 x vector<[3]xi64>>

// -----

func.func @outerproduct_add(%arg0: vector<2xf32>, %arg1: vector<3xf32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: @outerproduct_add(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>,
// CHECK-SAME:  %[[B:.*]]: vector<3xf32>,
// CHECK-SAME:  %[[C:.*]]: vector<2x3xf32>) -> vector<2x3xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[C]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T3:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:       %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T3]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T5:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T4]] : i64] : vector<2xf32>
// CHECK:       %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
// CHECK:       %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
// CHECK:       %[[T8:.*]] = llvm.extractvalue %[[T7]][0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T9:.*]] = llvm.intr.fmuladd(%[[T6]], %[[B]], %[[T8]]) : (vector<3xf32>, vector<3xf32>, vector<3xf32>) -> vector<3xf32>
// CHECK:       %[[T11:.*]] = llvm.insertvalue %[[T9]], %[[T10]][0] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T12:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[T13:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T12]] : i64] : vector<2xf32>
// CHECK:       %[[T14Insert:.*]] = llvm.insertelement %[[T13]]
// CHECK:       %[[T14:.*]] = llvm.shufflevector %[[T14Insert]]
// CHECK:       %[[T16:.*]] = llvm.extractvalue %[[T7]][1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T17:.*]] = llvm.intr.fmuladd(%[[T14]], %[[B]], %[[T16]]) : (vector<3xf32>, vector<3xf32>, vector<3xf32>) -> vector<3xf32>
// CHECK:       %[[T18:.*]] = llvm.insertvalue %[[T17]], %[[T11]][1] : !llvm.array<2 x vector<3xf32>>
// CHECK:       %[[T19:.*]] = builtin.unrealized_conversion_cast %[[T18]] : !llvm.array<2 x vector<3xf32>> to vector<2x3xf32>
// CHECK:       return %[[T19]] : vector<2x3xf32>

// -----

func.func @outerproduct_add_scalable(%arg0: vector<2xf32>, %arg1: vector<[3]xf32>, %arg2: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<[3]xf32>
  return %2 : vector<2x[3]xf32>
}
// CHECK-LABEL: @outerproduct_add_scalable
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>,
// CHECK-SAME:  %[[B:.*]]: vector<[3]xf32>,
// CHECK-SAME:  %[[C:.*]]: vector<2x[3]xf32>) -> vector<2x[3]xf32>
// CHECK:       %[[T7:.*]] = builtin.unrealized_conversion_cast %[[C]] : vector<2x[3]xf32> to !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T3:.*]] = arith.constant dense<0.000000e+00> : vector<2x[3]xf32>
// CHECK:       %[[T10:.*]] = builtin.unrealized_conversion_cast %[[T3]] : vector<2x[3]xf32> to !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[T5:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T4]] : i64] : vector<2xf32>
// CHECK:       %[[T6Insert:.*]] = llvm.insertelement %[[T5]]
// CHECK:       %[[T6:.*]] = llvm.shufflevector %[[T6Insert]]
// CHECK:       %[[T8:.*]] = llvm.extractvalue %[[T7]][0] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T9:.*]] = llvm.intr.fmuladd(%[[T6]], %[[B]], %[[T8]]) : (vector<[3]xf32>, vector<[3]xf32>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK:       %[[T11:.*]] = llvm.insertvalue %[[T9]], %[[T10]][0] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T12:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[T13:.*]] = llvm.extractelement %[[A]]{{\[}}%[[T12]] : i64] : vector<2xf32>
// CHECK:       %[[T14Insert:.*]] = llvm.insertelement %[[T13]]
// CHECK:       %[[T14:.*]] = llvm.shufflevector %[[T14Insert]]
// CHECK:       %[[T16:.*]] = llvm.extractvalue %[[T7]][1] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T17:.*]] = llvm.intr.fmuladd(%[[T14]], %[[B]], %[[T16]]) : (vector<[3]xf32>, vector<[3]xf32>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK:       %[[T18:.*]] = llvm.insertvalue %[[T17]], %[[T11]][1] : !llvm.array<2 x vector<[3]xf32>>
// CHECK:       %[[T19:.*]] = builtin.unrealized_conversion_cast %[[T18]] : !llvm.array<2 x vector<[3]xf32>> to vector<2x[3]xf32>
// CHECK:       return %[[T19]] : vector<2x[3]xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.mask { vector.outerproduct }
//===----------------------------------------------------------------------===//

func.func @masked_float_add_outerprod(%arg0: vector<2xf32>, %arg1: f32, %arg2: vector<2xf32>, %m: vector<2xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<add>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_float_add_outerprod(
// CHECK-SAME:                                          %[[VAL_0:.*]]: vector<2xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<2xf32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xf32> {
// CHECK:           %[[VAL_8:.*]] = llvm.intr.fmuladd(%[[VAL_0]], %{{.*}}, %[[VAL_2]])  : (vector<2xf32>, vector<2xf32>, vector<2xf32>) -> vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_3]], %[[VAL_8]], %[[VAL_2]] : vector<2xi1>, vector<2xf32>

// -----

func.func @masked_float_add_outerprod_scalable(%arg0: vector<[2]xf32>, %arg1: f32, %arg2: vector<[2]xf32>, %m: vector<[2]xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<add>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// CHECK-LABEL:   func.func @masked_float_add_outerprod_scalable(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: vector<[2]xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<[2]xf32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xf32> {
// CHECK:           %[[VAL_8:.*]] = llvm.intr.fmuladd(%[[VAL_0]], %{{.*}}, %[[VAL_2]])  : (vector<[2]xf32>, vector<[2]xf32>, vector<[2]xf32>) -> vector<[2]xf32>
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_3]], %[[VAL_8]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xf32>

// -----

func.func @masked_float_mul_outerprod(%arg0: vector<2xf32>, %arg1: f32, %arg2: vector<2xf32>, %m: vector<2xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<mul>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_float_mul_outerprod(
// CHECK-SAME:                                          %[[VAL_0:.*]]: vector<2xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<2xf32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xf32> {
// CHECK:           %[[VAL_8:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = arith.mulf %[[VAL_8]], %[[VAL_2]] : vector<2xf32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xf32>

// -----

func.func @masked_float_mul_outerprod_scalable(%arg0: vector<[2]xf32>, %arg1: f32, %arg2: vector<[2]xf32>, %m: vector<[2]xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<mul>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// CHECK-LABEL:   func.func @masked_float_mul_outerprod_scalable(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: vector<[2]xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<[2]xf32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xf32> {
// CHECK:           %[[VAL_8:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<[2]xf32>
// CHECK:           %[[VAL_9:.*]] = arith.mulf %[[VAL_8]], %[[VAL_2]] : vector<[2]xf32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xf32>

// -----

func.func @masked_float_max_outerprod(%arg0: vector<2xf32>, %arg1: f32, %arg2: vector<2xf32>, %m: vector<2xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<maxnumf>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_float_max_outerprod(
// CHECK-SAME:                                          %[[VAL_0:.*]]: vector<2xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<2xf32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xf32> {
// CHECK:           %[[VAL_8:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = arith.maxnumf %[[VAL_8]], %[[VAL_2]] : vector<2xf32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xf32>

// -----

func.func @masked_float_max_outerprod_scalable(%arg0: vector<[2]xf32>, %arg1: f32, %arg2: vector<[2]xf32>, %m: vector<[2]xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<maxnumf>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// CHECK-LABEL:   func.func @masked_float_max_outerprod_scalable(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: vector<[2]xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<[2]xf32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xf32> {
// CHECK:           %[[VAL_8:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<[2]xf32>
// CHECK:           %[[VAL_9:.*]] = arith.maxnumf %[[VAL_8]], %[[VAL_2]] : vector<[2]xf32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xf32>

// -----

func.func @masked_float_min_outerprod(%arg0: vector<2xf32>, %arg1: f32, %arg2: vector<2xf32>, %m: vector<2xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<minnumf>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_float_min_outerprod(
// CHECK-SAME:                                          %[[VAL_0:.*]]: vector<2xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<2xf32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xf32> {
// CHECK:           %[[VAL_8:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = arith.minnumf %[[VAL_8]], %[[VAL_2]] : vector<2xf32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xf32>

// -----

func.func @masked_float_min_outerprod_scalable(%arg0: vector<[2]xf32>, %arg1: f32, %arg2: vector<[2]xf32>, %m: vector<[2]xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<minnumf>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// CHECK-LABEL:   func.func @masked_float_min_outerprod_scalable(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: vector<[2]xf32>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: vector<[2]xf32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xf32> {
// CHECK:           %[[VAL_8:.*]] = arith.mulf %[[VAL_0]], %{{.*}} : vector<[2]xf32>
// CHECK:           %[[VAL_9:.*]] = arith.minnumf %[[VAL_8]], %[[VAL_2]] : vector<[2]xf32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xf32>

// -----

func.func @masked_int_add_outerprod(%arg0: vector<2xi32>, %arg1: i32, %arg2: vector<2xi32>, %m: vector<2xi1>) -> vector<2xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<add>} : vector<2xi32>, i32 } : vector<2xi1> -> vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL:   func.func @masked_int_add_outerprod(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<2xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<2xi32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_2]] : vector<2xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xi32>

// -----

func.func @masked_int_add_outerprod_scalable(%arg0: vector<[2]xi32>, %arg1: i32, %arg2: vector<[2]xi32>, %m: vector<[2]xi1>) -> vector<[2]xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<add>} : vector<[2]xi32>, i32 } : vector<[2]xi1> -> vector<[2]xi32>
  return %0 : vector<[2]xi32>
}

// CHECK-LABEL:   func.func @masked_int_add_outerprod_scalable(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: vector<[2]xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<[2]xi32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<[2]xi32>
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_2]] : vector<[2]xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xi32>

// -----

func.func @masked_int_mul_outerprod(%arg0: vector<2xi32>, %arg1: i32, %arg2: vector<2xi32>, %m: vector<2xi1>) -> vector<2xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<mul>} : vector<2xi32>, i32 } : vector<2xi1> -> vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL:   func.func @masked_int_mul_outerprod(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<2xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<2xi32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_8]], %[[VAL_2]] : vector<2xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xi32>

// -----

func.func @masked_int_mul_outerprod_scalable(%arg0: vector<[2]xi32>, %arg1: i32, %arg2: vector<[2]xi32>, %m: vector<[2]xi1>) -> vector<[2]xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<mul>} : vector<[2]xi32>, i32 } : vector<[2]xi1> -> vector<[2]xi32>
  return %0 : vector<[2]xi32>
}

// CHECK-LABEL:   func.func @masked_int_mul_outerprod_scalable(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: vector<[2]xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<[2]xi32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<[2]xi32>
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_8]], %[[VAL_2]] : vector<[2]xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xi32>

// -----

func.func @masked_int_max_outerprod(%arg0: vector<2xi32>, %arg1: i32, %arg2: vector<2xi32>, %m: vector<2xi1>) -> vector<2xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<maxsi>} : vector<2xi32>, i32 } : vector<2xi1> -> vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL:   func.func @masked_int_max_outerprod(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<2xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<2xi32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.maxsi %[[VAL_8]], %[[VAL_2]] : vector<2xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xi32>

// -----

func.func @masked_int_max_outerprod_scalable(%arg0: vector<[2]xi32>, %arg1: i32, %arg2: vector<[2]xi32>, %m: vector<[2]xi1>) -> vector<[2]xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<maxsi>} : vector<[2]xi32>, i32 } : vector<[2]xi1> -> vector<[2]xi32>
  return %0 : vector<[2]xi32>
}

// CHECK-LABEL:   func.func @masked_int_max_outerprod_scalable(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: vector<[2]xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<[2]xi32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<[2]xi32>
// CHECK:           %[[VAL_9:.*]] = arith.maxsi %[[VAL_8]], %[[VAL_2]] : vector<[2]xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xi32>

// -----

func.func @masked_int_min_outerprod(%arg0: vector<2xi32>, %arg1: i32, %arg2: vector<2xi32>, %m: vector<2xi1>) -> vector<2xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<minui>} : vector<2xi32>, i32 } : vector<2xi1> -> vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL:   func.func @masked_int_min_outerprod(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<2xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<2xi32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.minui %[[VAL_8]], %[[VAL_2]] : vector<2xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xi32>

// -----

func.func @masked_int_min_outerprod_scalable(%arg0: vector<[2]xi32>, %arg1: i32, %arg2: vector<[2]xi32>, %m: vector<[2]xi1>) -> vector<[2]xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<minui>} : vector<[2]xi32>, i32 } : vector<[2]xi1> -> vector<[2]xi32>
  return %0 : vector<[2]xi32>
}

// CHECK-LABEL:   func.func @masked_int_min_outerprod_scalable(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: vector<[2]xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<[2]xi32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<[2]xi32>
// CHECK:           %[[VAL_9:.*]] = arith.minui %[[VAL_8]], %[[VAL_2]] : vector<[2]xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xi32>

// -----

func.func @masked_int_and_outerprod(%arg0: vector<2xi32>, %arg1: i32, %arg2: vector<2xi32>, %m: vector<2xi1>) -> vector<2xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<and>} : vector<2xi32>, i32 } : vector<2xi1> -> vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL:   func.func @masked_int_and_outerprod(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<2xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<2xi32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.andi %[[VAL_8]], %[[VAL_2]] : vector<2xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xi32>

// -----

func.func @masked_int_and_outerprod_scalable(%arg0: vector<[2]xi32>, %arg1: i32, %arg2: vector<[2]xi32>, %m: vector<[2]xi1>) -> vector<[2]xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<and>} : vector<[2]xi32>, i32 } : vector<[2]xi1> -> vector<[2]xi32>
  return %0 : vector<[2]xi32>
}

// CHECK-LABEL:   func.func @masked_int_and_outerprod_scalable(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: vector<[2]xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<[2]xi32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<[2]xi32>
// CHECK:           %[[VAL_9:.*]] = arith.andi %[[VAL_8]], %[[VAL_2]] : vector<[2]xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xi32>

// -----

func.func @masked_int_or_outerprod(%arg0: vector<2xi32>, %arg1: i32, %arg2: vector<2xi32>, %m: vector<2xi1>) -> vector<2xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<or>} : vector<2xi32>, i32 } : vector<2xi1> -> vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL:   func.func @masked_int_or_outerprod(
// CHECK-SAME:                                       %[[VAL_0:.*]]: vector<2xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<2xi32>, %[[VAL_3:.*]]: vector<2xi1>) -> vector<2xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<2xi32>
// CHECK:           %[[VAL_9:.*]] = arith.ori %[[VAL_8]], %[[VAL_2]] : vector<2xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<2xi1>, vector<2xi32>

// -----

func.func @masked_int_or_outerprod_scalable(%arg0: vector<[2]xi32>, %arg1: i32, %arg2: vector<[2]xi32>, %m: vector<[2]xi1>) -> vector<[2]xi32> {
  %0 = vector.mask %m { vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<or>} : vector<[2]xi32>, i32 } : vector<[2]xi1> -> vector<[2]xi32>
  return %0 : vector<[2]xi32>
}

// CHECK-LABEL:   func.func @masked_int_or_outerprod_scalable
// CHECK-SAME:                                       %[[VAL_0:.*]]: vector<[2]xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: vector<[2]xi32>, %[[VAL_3:.*]]: vector<[2]xi1>) -> vector<[2]xi32> {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_0]], %{{.*}} : vector<[2]xi32>
// CHECK:           %[[VAL_9:.*]] = arith.ori %[[VAL_8]], %[[VAL_2]] : vector<[2]xi32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_3]], %[[VAL_9]], %[[VAL_2]] : vector<[2]xi1>, vector<[2]xi32>

// -----

//===----------------------------------------------------------------------===//
// vector.extract
//===----------------------------------------------------------------------===//

// FIXME: Segfaults for --convert-to-llvm="filter-dialects=vector"
func.func @extract_scalar_from_vec_1d_f32_poison_idx(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.extract %arg0[-1]: f32 from vector<16xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_f32_poison_idx
//       CHECK:   %[[UB:.*]] = ub.poison : f32
//       CHECK:   return %[[UB]] : f32

// -----

// FIXME: Segfaults for --convert-to-llvm="filter-dialects=vector"
func.func @extract_vec_2d_from_vec_3d_f32_poison_idx(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extract %arg0[-1]: vector<3x16xf32> from vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: @extract_vec_2d_from_vec_3d_f32_poison_idx
//       CHECK:   %[[UB:.*]] = ub.poison : vector<3x16xf32>
//       CHECK:   return %[[UB]] : vector<3x16xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.print
//===----------------------------------------------------------------------===//

func.func @print_scalar_i1(%arg0: i1) {
  vector.print %arg0 : i1
  return
}
//
// Type "boolean" always uses zero extension.
//
// CHECK-LABEL: @print_scalar_i1(
// CHECK-SAME: %[[A:.*]]: i1)
//       CHECK: %[[S:.*]] = arith.extui %[[A]] : i1 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_i4(%arg0: i4) {
  vector.print %arg0 : i4
  return
}
// CHECK-LABEL: @print_scalar_i4(
// CHECK-SAME: %[[A:.*]]: i4)
//       CHECK: %[[S:.*]] = arith.extsi %[[A]] : i4 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_si4(%arg0: si4) {
  vector.print %arg0 : si4
  return
}
// CHECK-LABEL: @print_scalar_si4(
// CHECK-SAME: %[[A:.*]]: si4)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : si4 to i4
//       CHECK: %[[S:.*]] = arith.extsi %[[C]] : i4 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_ui4(%arg0: ui4) {
  vector.print %arg0 : ui4
  return
}
// CHECK-LABEL: @print_scalar_ui4(
// CHECK-SAME: %[[A:.*]]: ui4)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui4 to i4
//       CHECK: %[[S:.*]] = arith.extui %[[C]] : i4 to i64
//       CHECK: llvm.call @printU64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_i32(%arg0: i32) {
  vector.print %arg0 : i32
  return
}
// CHECK-LABEL: @print_scalar_i32(
// CHECK-SAME: %[[A:.*]]: i32)
//       CHECK: %[[S:.*]] = arith.extsi %[[A]] : i32 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_ui32(%arg0: ui32) {
  vector.print %arg0 : ui32
  return
}
// CHECK-LABEL: @print_scalar_ui32(
// CHECK-SAME: %[[A:.*]]: ui32)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui32 to i32
//       CHECK: %[[S:.*]] = arith.extui %[[C]] : i32 to i64
//       CHECK: llvm.call @printU64(%[[S]]) : (i64) -> ()

// -----

func.func @print_scalar_i40(%arg0: i40) {
  vector.print %arg0 : i40
  return
}
// CHECK-LABEL: @print_scalar_i40(
// CHECK-SAME: %[[A:.*]]: i40)
//       CHECK: %[[S:.*]] = arith.extsi %[[A]] : i40 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_si40(%arg0: si40) {
  vector.print %arg0 : si40
  return
}
// CHECK-LABEL: @print_scalar_si40(
// CHECK-SAME: %[[A:.*]]: si40)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : si40 to i40
//       CHECK: %[[S:.*]] = arith.extsi %[[C]] : i40 to i64
//       CHECK: llvm.call @printI64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_ui40(%arg0: ui40) {
  vector.print %arg0 : ui40
  return
}
// CHECK-LABEL: @print_scalar_ui40(
// CHECK-SAME: %[[A:.*]]: ui40)
//       CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui40 to i40
//       CHECK: %[[S:.*]] = arith.extui %[[C]] : i40 to i64
//       CHECK: llvm.call @printU64(%[[S]]) : (i64) -> ()
//       CHECK: llvm.call @printNewline() : () -> ()

// -----

//===----------------------------------------------------------------------===//
// vector.extract_strided_slice
//===----------------------------------------------------------------------===//

func.func @extract_strided_slice_f32_1d_from_1d(%arg0: vector<4xf32>) -> vector<2xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: @extract_strided_slice_f32_1d_from_1d
//  CHECK-SAME:    %[[A:.*]]: vector<4xf32>)
//       CHECK:    %[[T0:.*]] = llvm.shufflevector %[[A]], %[[A]] [2, 3] : vector<4xf32>
//       CHECK:    return %[[T0]] : vector<2xf32>

// NOTE: For scalable vectors we could only extract vector<[4]xf32> from vector<[4]xf32>, but that would be a NOP.

// -----

func.func @extract_strided_slice_index_1d_from_1d(%arg0: vector<4xindex>) -> vector<2xindex> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xindex> to vector<2xindex>
  return %0 : vector<2xindex>
}
// CHECK-LABEL: @extract_strided_slice_index_1d_from_1d
//  CHECK-SAME:    %[[A:.*]]: vector<4xindex>)
//       CHECK:    %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4xindex> to vector<4xi64>
//       CHECK:    %[[T2:.*]] = llvm.shufflevector %[[T0]], %[[T0]] [2, 3] : vector<4xi64>
//       CHECK:    %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : vector<2xi64> to vector<2xindex>
//       CHECK:    return %[[T3]] : vector<2xindex>

// NOTE: For scalable vectors we could only extract vector<[4]xindex> from vector<[4]xindex>, but that would be a NOP.

// -----

func.func @extract_strided_slice_f32_1d_from_2d(%arg0: vector<4x8xf32>) -> vector<2x8xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4x8xf32> to vector<2x8xf32>
  return %0 : vector<2x8xf32>
}
// CHECK-LABEL: @extract_strided_slice_f32_1d_from_2d(
//  CHECK-SAME:    %[[ARG:.*]]: vector<4x8xf32>)
//       CHECK:    %[[A:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x8xf32> to !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T0:.*]] = llvm.mlir.poison : !llvm.array<2 x vector<8xf32>>
//       CHECK:    %[[T1:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T2:.*]] = llvm.insertvalue %[[T1]], %[[T0]][0] : !llvm.array<2 x vector<8xf32>>
//       CHECK:    %[[T3:.*]] = llvm.extractvalue %[[A]][3] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T4:.*]] = llvm.insertvalue %[[T3]], %[[T2]][1] : !llvm.array<2 x vector<8xf32>>
//       CHECK:    %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : !llvm.array<2 x vector<8xf32>> to vector<2x8xf32>
//       CHECK:    return %[[T5]]

// -----

func.func @extract_strided_slice_f32_1d_from_2d_scalable(%arg0: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4x[8]xf32> to vector<2x[8]xf32>
  return %0 : vector<2x[8]xf32>
}
// CHECK-LABEL:   func.func @extract_strided_slice_f32_1d_from_2d_scalable(
//  CHECK-SAME:    %[[ARG:.*]]: vector<4x[8]xf32>)
//       CHECK:    %[[A:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x[8]xf32> to !llvm.array<4 x vector<[8]xf32>>
//       CHECK:    %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x[8]xf32>
//       CHECK:    %[[DST:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<2x[8]xf32> to !llvm.array<2 x vector<[8]xf32>>
//       CHECK:    %[[E0:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<4 x vector<[8]xf32>>
//       CHECK:    %[[E1:.*]] = llvm.extractvalue %[[A]][3] : !llvm.array<4 x vector<[8]xf32>>
//       CHECK:    %[[I0:.*]] = llvm.insertvalue %[[E0]], %[[DST]][0] : !llvm.array<2 x vector<[8]xf32>>
//       CHECK:    %[[I1:.*]] = llvm.insertvalue %[[E1]], %[[I0]][1] : !llvm.array<2 x vector<[8]xf32>>
//       CHECK:    %[[RES:.*]] = builtin.unrealized_conversion_cast %[[I1]] : !llvm.array<2 x vector<[8]xf32>> to vector<2x[8]xf32>
//       CHECK:    return %[[RES]]

// -----

func.func @extract_strided_slice_f32_2d_from_2d(%arg0: vector<4x8xf32>) -> vector<2x2xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8xf32> to vector<2x2xf32>
  return %0 : vector<2x2xf32>
}
// CHECK-LABEL: @extract_strided_slice_f32_2d_from_2d(
//  CHECK-SAME:    %[[ARG:.*]]: vector<4x8xf32>)
//       CHECK:    %[[A:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x8xf32> to !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
//       CHECK:    %[[VAL_6:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : vector<2x2xf32> to !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[T2:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T3:.*]] = llvm.shufflevector %[[T2]], %[[T2]] [2, 3] : vector<8xf32>
//       CHECK:    %[[T4:.*]] = llvm.insertvalue %[[T3]], %[[VAL_6]][0] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[T5:.*]] = llvm.extractvalue %[[A]][3] : !llvm.array<4 x vector<8xf32>>
//       CHECK:    %[[T6:.*]] = llvm.shufflevector %[[T5]], %[[T5]] [2, 3] : vector<8xf32>
//       CHECK:    %[[T7:.*]] = llvm.insertvalue %[[T6]], %[[T4]][1] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[T7]] : !llvm.array<2 x vector<2xf32>> to vector<2x2xf32>
//       CHECK:    return %[[VAL_12]] : vector<2x2xf32>

// -----

// NOTE: For scalable vectors, we can only extract "full" scalable dimensions
// (e.g. [8] from [8], but not [4] from [8]).

func.func @extract_strided_slice_f32_2d_from_2d_scalable(%arg0: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x[8]xf32> to vector<2x[8]xf32>
  return %0 : vector<2x[8]xf32>
}
// CHECK-LABEL: @extract_strided_slice_f32_2d_from_2d_scalable(
//  CHECK-SAME:     %[[ARG:.*]]: vector<4x[8]xf32>)
// CHECK:           %[[T1:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x[8]xf32> to !llvm.array<4 x vector<[8]xf32>>
// CHECK:           %[[T3:.*]] = arith.constant dense<0.000000e+00> : vector<2x[8]xf32>
// CHECK:           %[[T4:.*]] = builtin.unrealized_conversion_cast %[[T3]] : vector<2x[8]xf32> to !llvm.array<2 x vector<[8]xf32>>
// CHECK:           %[[T5:.*]] = llvm.extractvalue %[[T1]][2] : !llvm.array<4 x vector<[8]xf32>>
// CHECK:           %[[T6:.*]] = llvm.insertvalue %[[T5]], %[[T4]][0] : !llvm.array<2 x vector<[8]xf32>>
// CHECK:           %[[T7:.*]] = llvm.extractvalue %[[T1]][3] : !llvm.array<4 x vector<[8]xf32>>
// CHECK:           %[[T8:.*]] = llvm.insertvalue %[[T7]], %[[T6]][1] : !llvm.array<2 x vector<[8]xf32>>
// CHECK:           %[[T9:.*]] = builtin.unrealized_conversion_cast %[[T8]] : !llvm.array<2 x vector<[8]xf32>> to vector<2x[8]xf32>
// CHECK:           return %[[T9]] : vector<2x[8]xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.insert_strided_slice
//===----------------------------------------------------------------------===//

func.func @insert_strided_slice_f32_2d_into_3d(%b: vector<4x4xf32>, %c: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x4xf32> into vector<4x4x4xf32>
  return %0 : vector<4x4x4xf32>
}
// CHECK-LABEL: @insert_strided_slice_f32_2d_into_3d
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.array<4 x array<4 x vector<4xf32>>>

// -----

func.func @insert_strided_slice_f32_2d_into_3d_scalable(%b: vector<4x[4]xf32>, %c: vector<4x4x[4]xf32>) -> vector<4x4x[4]xf32> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x[4]xf32> into vector<4x4x[4]xf32>
  return %0 : vector<4x4x[4]xf32>
}
// CHECK-LABEL: @insert_strided_slice_f32_2d_into_3d_scalable
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.array<4 x array<4 x vector<[4]xf32>>>

// -----

func.func @insert_strided_index_slice_index_2d_into_3d(%b: vector<4x4xindex>, %c: vector<4x4x4xindex>) -> vector<4x4x4xindex> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x4xindex> into vector<4x4x4xindex>
  return %0 : vector<4x4x4xindex>
}
// CHECK-LABEL: @insert_strided_index_slice_index_2d_into_3d
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.array<4 x array<4 x vector<4xi64>>>

// -----

func.func @insert_strided_index_slice_index_2d_into_3d_scalable(%b: vector<4x[4]xindex>, %c: vector<4x4x[4]xindex>) -> vector<4x4x[4]xindex> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x[4]xindex> into vector<4x4x[4]xindex>
  return %0 : vector<4x4x[4]xindex>
}
// CHECK-LABEL: @insert_strided_index_slice_index_2d_into_3d_scalable
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.array<4 x array<4 x vector<[4]xi64>>>

// -----

func.func @insert_strided_slice_f32_2d_into_2d(%a: vector<2x2xf32>, %b: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: @insert_strided_slice_f32_2d_into_2d
//
// Subvector vector<2xf32> @0 into vector<4xf32> @2
//       CHECK:    %[[V2_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[V4_0:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.array<4 x vector<4xf32>>
// Element @0 -> element @2
//       CHECK:    %[[R4_0:.*]] = llvm.shufflevector %[[V2_0]], %[[V2_0]] [0, 1, 0, 0] : vector<2xf32>
//       CHECK:    %[[R4_1:.*]] = llvm.shufflevector %[[R4_0]], %[[V4_0]] [4, 5, 0, 1] : vector<4xf32>
//       CHECK:    llvm.insertvalue %[[R4_1]], {{.*}}[2] : !llvm.array<4 x vector<4xf32>>
//
// Subvector vector<2xf32> @1 into vector<4xf32> @3
//       CHECK:    %[[V2_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.array<2 x vector<2xf32>>
//       CHECK:    %[[V4_3:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.array<4 x vector<4xf32>>
// Element @0 -> element @2
//       CHECK:    %[[R4_2:.*]] = llvm.shufflevector %[[V2_1]], %[[V2_1]] [0, 1, 0, 0] : vector<2xf32>
//       CHECK:    %[[R4_3:.*]] = llvm.shufflevector %[[R4_2]], %[[V4_3]] [4, 5, 0, 1] : vector<4xf32>
//       CHECK:    llvm.insertvalue %[[R4_3]], {{.*}}[3] : !llvm.array<4 x vector<4xf32>>

// -----

// NOTE: For scalable dimensions, the corresponding "base" size must match
// (i.e. we can only insert "full" scalable dimensions, e.g. [2] into [2], but
// not [2] from [4]).

func.func @insert_strided_slice_f32_2d_into_2d_scalable(%a: vector<2x[2]xf32>, %b: vector<4x[2]xf32>) -> vector<4x[2]xf32> {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 0], strides = [1, 1]} : vector<2x[2]xf32> into vector<4x[2]xf32>
  return %0 : vector<4x[2]xf32>
}

// CHECK-LABEL:   func.func @insert_strided_slice_f32_2d_into_2d_scalable
// Subvector vector<[2]xf32> @0 into vector<[4]xf32> @2
// CHECK:           %[[A_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.array<2 x vector<[2]xf32>>
// Element @0 -> element @2
// CHECK:           %[[B_UPDATED:.*]] = llvm.insertvalue %[[A_0]], {{.*}}[2] : !llvm.array<4 x vector<[2]xf32>>
// Subvector vector<[2]xf32> @1 into vector<[4]xf32> @3
// CHECK:           %[[A_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.array<2 x vector<[2]xf32>>
// Element @0 -> element @2
// CHECK:           llvm.insertvalue %[[A_1]], %[[B_UPDATED]][3] : !llvm.array<4 x vector<[2]xf32>>

// -----

func.func @insert_strided_slice_f32_2d_into_3d(%arg0: vector<2x4xf32>, %arg1: vector<16x4x8xf32>) -> vector<16x4x8xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 2], strides = [1, 1]}:
        vector<2x4xf32> into vector<16x4x8xf32>
  return %0 : vector<16x4x8xf32>
}
// CHECK-LABEL: func @insert_strided_slice_f32_2d_into_3d
//       CHECK:    %[[V4_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.array<2 x vector<4xf32>>
//       CHECK:    %[[V4_0_0:.*]] = llvm.extractvalue {{.*}}[0, 0] : !llvm.array<16 x array<4 x vector<8xf32>>>
//       CHECK:    %[[R8_0:.*]] = llvm.shufflevector %[[V4_0]], %[[V4_0]] [0, 1, 2, 3, 0, 0, 0, 0] : vector<4xf32>
//       CHECK:    %[[R8_1:.*]] = llvm.shufflevector %[[R8_0:.*]], %[[V4_0_0]] [8, 9, 0, 1, 2, 3, 14, 15] : vector<8xf32>
//       CHECK:    llvm.insertvalue %[[R8_1]], {{.*}}[0] : !llvm.array<4 x vector<8xf32>>

//       CHECK:    %[[V4_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.array<2 x vector<4xf32>>
//       CHECK:    %[[V4_0_1:.*]] = llvm.extractvalue {{.*}}[0, 1] : !llvm.array<16 x array<4 x vector<8xf32>>>
//       CHECK:    %[[R8_2:.*]] = llvm.shufflevector %[[V4_1]], %[[V4_1]] [0, 1, 2, 3, 0, 0, 0, 0] : vector<4xf32>
//       CHECK:    %[[R8_3:.*]] = llvm.shufflevector %[[R8_2]], %[[V4_0_1]] [8, 9, 0, 1, 2, 3, 14, 15] : vector<8xf32>
//       CHECK:    llvm.insertvalue %[[R8_3]], {{.*}}[1] : !llvm.array<4 x vector<8xf32>>

// -----

// NOTE: For scalable dimensions, the corresponding "base" size must match
// (i.e. we can only insert "full" scalable dimensions, e.g. [4] into [4], but
// not [4] from [8]).

func.func @insert_strided_slice_f32_2d_into_3d_scalable(%arg0: vector<2x[4]xf32>, %arg1: vector<16x4x[4]xf32>) -> vector<16x4x[4]xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [3, 2, 0], strides = [1, 1]}:
        vector<2x[4]xf32> into vector<16x4x[4]xf32>
  return %0 : vector<16x4x[4]xf32>
}

// CHECK-LABEL:   func.func @insert_strided_slice_f32_2d_into_3d_scalable(

// Subvector vector<4x[4]xf32> from vector<16x4x[4]xf32> @3
// CHECK:           %[[ARG_1_0:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.array<16 x array<4 x vector<[4]xf32>>>

// Subvector vector<[4]xf32> @0 into vector<4x[4]xf32> @2
// CHECK:           %[[ARG_0_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.array<2 x vector<[4]xf32>>
// CHECK:           %[[B_UPDATED_0:.*]] = llvm.insertvalue %[[ARG_0_0]], %[[ARG_1_0]][2] : !llvm.array<4 x vector<[4]xf32>>

// Subvector vector<[4]xf32> @1 into vector<4x[4]xf32> @3
// CHECK:           %[[ARG_0_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.array<2 x vector<[4]xf32>>
// CHECK:           %[[B_UPDATED_1:.*]] = llvm.insertvalue %[[ARG_0_1]], %[[B_UPDATED_0]][3] : !llvm.array<4 x vector<[4]xf32>>

// Subvector vector<4x[4]xf32> into vector<16x4x[4]xf32> @3
// CHECK:           llvm.insertvalue %[[B_UPDATED_1]], {{.*}}[3] : !llvm.array<16 x array<4 x vector<[4]xf32>>>

// -----

//===----------------------------------------------------------------------===//
// vector.fma
//===----------------------------------------------------------------------===//

func.func @fma(%vec_1d: vector<8xf32>, %vec_2d: vector<2x4xf32>, %vec_3d: vector<1x1x1xf32>, %vec_0d: vector<f32>) -> (vector<8xf32>, vector<2x4xf32>, vector<1x1x1xf32>, vector<f32>) {
  // CHECK-LABEL: @fma
  //  CHECK-SAME: %[[VEC_1D:.*]]: vector<8xf32>
  //  CHECK-SAME: %[[VEC_2D:.*]]: vector<2x4xf32>
  //  CHECK-SAME: %[[VEC_3D:.*]]: vector<1x1x1xf32>
  //       CHECK: %[[VEC_2D_CAST:.*]] = builtin.unrealized_conversion_cast %[[VEC_2D]] : vector<2x4xf32> to !llvm.array<2 x vector<4xf32>>
  //       CHECK: llvm.intr.fmuladd
  //  CHECK-SAME:   (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %0 = vector.fma %vec_1d, %vec_1d, %vec_1d : vector<8xf32>

  //       CHECK: %[[VEC_2D_00:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_01:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_02:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_ADD_1:.*]] = llvm.intr.fmuladd(%[[VEC_2D_00]], %[[VEC_2D_01]], %[[VEC_2D_02]]) :
  //  CHECK-SAME: (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  //       CHECK: llvm.insertvalue %[[VEC_2D_ADD_1]], {{.*}}[0] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_10:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][1] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_11:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][1] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_12:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][1] : !llvm.array<2 x vector<4xf32>>
  //       CHECK: %[[VEC_2D_ADD_2:.*]] = llvm.intr.fmuladd(%[[VEC_2D_10]], %[[VEC_2D_11]], %[[VEC_2D_12]]) :
  //  CHECK-SAME: (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  //       CHECK: llvm.insertvalue %[[VEC_2D_ADD_2]], {{.*}}[1] : !llvm.array<2 x vector<4xf32>>
  %1 = vector.fma %vec_2d, %vec_2d, %vec_2d : vector<2x4xf32>

  //       CHECK: %[[C0:.*]] = llvm.intr.fmuladd
  //  CHECK-SAME:   (vector<1xf32>, vector<1xf32>, vector<1xf32>) -> vector<1xf32>
  %2 = vector.fma %vec_3d, %vec_3d, %vec_3d : vector<1x1x1xf32>

  //       CHECK: %[[D0:.*]] = llvm.intr.fmuladd
  //  CHECK-SAME:   (vector<1xf32>, vector<1xf32>, vector<1xf32>) -> vector<1xf32>
  %3 = vector.fma %vec_0d, %vec_0d, %vec_0d : vector<f32>

  return %0, %1, %2, %3: vector<8xf32>, vector<2x4xf32>, vector<1x1x1xf32>, vector<f32>
}

// -----

func.func @fma_scalable(%vec_1d: vector<[8]xf32>, %vec_2d: vector<2x[4]xf32>, %vec_3d: vector<1x1x[1]xf32>, %vec_0d: vector<f32>) -> (vector<[8]xf32>, vector<2x[4]xf32>, vector<1x1x[1]xf32>) {
  // CHECK-LABEL: @fma_scalable
  //  CHECK-SAME: %[[VEC_1D:.*]]: vector<[8]xf32>
  //  CHECK-SAME: %[[VEC_2D:.*]]: vector<2x[4]xf32>
  //  CHECK-SAME: %[[VEC_3D:.*]]: vector<1x1x[1]xf32>
  //       CHECK: %[[VEC_2D_CAST:.*]] = builtin.unrealized_conversion_cast %[[VEC_2D]] : vector<2x[4]xf32> to !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: llvm.intr.fmuladd
  //  CHECK-SAME:   (vector<[8]xf32>, vector<[8]xf32>, vector<[8]xf32>) -> vector<[8]xf32>
  %0 = vector.fma %vec_1d, %vec_1d, %vec_1d : vector<[8]xf32>

  //       CHECK: %[[VEC_2D_00:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][0] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_01:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][0] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_02:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][0] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_ADD_1:.*]] = llvm.intr.fmuladd(%[[VEC_2D_00]], %[[VEC_2D_01]], %[[VEC_2D_02]]) :
  //  CHECK-SAME: (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
  //       CHECK: llvm.insertvalue %[[VEC_2D_ADD_1]], {{.*}}[0] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_10:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][1] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_11:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][1] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_12:.*]] = llvm.extractvalue %[[VEC_2D_CAST]][1] : !llvm.array<2 x vector<[4]xf32>>
  //       CHECK: %[[VEC_2D_ADD_2:.*]] = llvm.intr.fmuladd(%[[VEC_2D_10]], %[[VEC_2D_11]], %[[VEC_2D_12]]) :
  //  CHECK-SAME: (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
  //       CHECK: llvm.insertvalue %[[VEC_2D_ADD_2]], {{.*}}[1] : !llvm.array<2 x vector<[4]xf32>>
  %1 = vector.fma %vec_2d, %vec_2d, %vec_2d : vector<2x[4]xf32>

  //       CHECK: %[[C0:.*]] = llvm.intr.fmuladd
  //  CHECK-SAME:   (vector<[1]xf32>, vector<[1]xf32>, vector<[1]xf32>) -> vector<[1]xf32>
  %2 = vector.fma %vec_3d, %vec_3d, %vec_3d : vector<1x1x[1]xf32>

  return %0, %1, %2: vector<[8]xf32>, vector<2x[4]xf32>, vector<1x1x[1]xf32>
}

// -----

//===----------------------------------------------------------------------===//
// vector.constant_mask
//===----------------------------------------------------------------------===//

func.func @constant_mask_0d_f() -> vector<i1> {
  %0 = vector.constant_mask [0] : vector<i1>
  return %0 : vector<i1>
}
// CHECK-LABEL: func @constant_mask_0d_f
// CHECK: %[[VAL_0:.*]] = arith.constant dense<false> : vector<i1>
// CHECK: return %[[VAL_0]] : vector<i1>

// -----

func.func @constant_mask_0d_t() -> vector<i1> {
  %0 = vector.constant_mask [1] : vector<i1>
  return %0 : vector<i1>
}
// CHECK-LABEL: func @constant_mask_0d_t
// CHECK: %[[VAL_0:.*]] = arith.constant dense<true> : vector<i1>
// CHECK: return %[[VAL_0]] : vector<i1>

// -----

func.func @constant_mask_1d() -> vector<8xi1> {
  %0 = vector.constant_mask [4] : vector<8xi1>
  return %0 : vector<8xi1>
}
// CHECK-LABEL: func @constant_mask_1d
// CHECK: %[[VAL_0:.*]] = arith.constant dense<[true, true, true, true, false, false, false, false]> : vector<8xi1>
// CHECK: return %[[VAL_0]] : vector<8xi1>

// -----

func.func @constant_mask_1d_scalable_all_false() -> vector<[8]xi1> {
  %0 = vector.constant_mask [0] : vector<[8]xi1>
  return %0 : vector<[8]xi1>
}
// CHECK-LABEL: func @constant_mask_1d_scalable_all_false
// CHECK: %[[VAL_0:.*]] = arith.constant dense<false> : vector<[8]xi1>
// CHECK: return %[[VAL_0]] : vector<[8]xi1>

// -----

func.func @constant_mask_1d_scalable_all_true() -> vector<[8]xi1> {
  %0 = vector.constant_mask [8] : vector<[8]xi1>
  return %0 : vector<[8]xi1>
}
// CHECK-LABEL: func @constant_mask_1d_scalable_all_true
// CHECK: %[[VAL_0:.*]] = arith.constant dense<true> : vector<[8]xi1>
// CHECK: return %[[VAL_0]] : vector<[8]xi1>

// -----

func.func @constant_mask_2d() -> vector<4x4xi1> {
  %v = vector.constant_mask [2, 2] : vector<4x4xi1>
  return %v: vector<4x4xi1>
}

// CHECK-LABEL: func @constant_mask_2d
// CHECK: %[[VAL_0:.*]] = arith.constant 
// CHECK-SAME{LITERAL}: dense<[[true, true, false, false], [true, true, false, false], [false, false, false, false], [false, false, false, false]]> : vector<4x4xi1>
// CHECK: return %[[VAL_0]] : vector<4x4xi1>

// -----

func.func @constant_mask_2d_trailing_scalable() -> vector<4x[4]xi1> {
  %0 = vector.constant_mask [2, 4] : vector<4x[4]xi1>
  return %0 : vector<4x[4]xi1>
}
// CHECK-LABEL:   func.func @constant_mask_2d_trailing_scalable
// CHECK:           %[[VAL_0:.*]] = arith.constant dense<true> : vector<[4]xi1>
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<false> : vector<4x[4]xi1>
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<4x[4]xi1> to !llvm.array<4 x vector<[4]xi1>>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][0] : !llvm.array<4 x vector<[4]xi1>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][1] : !llvm.array<4 x vector<[4]xi1>>
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : !llvm.array<4 x vector<[4]xi1>> to vector<4x[4]xi1>
// CHECK:           return %[[VAL_5]] : vector<4x[4]xi1>

// -----

/// Currently, this is not supported as generating the mask would require
/// unrolling the leading scalable dimension at compile time.
func.func @negative_constant_mask_2d_leading_scalable() -> vector<[4]x4xi1> {
  %0 = vector.constant_mask [4, 2] : vector<[4]x4xi1>
  return %0 : vector<[4]x4xi1>
}
// CHECK-LABEL:   func.func @negative_constant_mask_2d_leading_scalable
// CHECK:           %[[VAL_0:.*]] = vector.constant_mask [4, 2] : vector<[4]x4xi1>
// CHECK:           return %[[VAL_0]] : vector<[4]x4xi1>

// -----

//===----------------------------------------------------------------------===//
// vector.create_mask
//===----------------------------------------------------------------------===//

func.func @create_mask_0d(%num_elems : index) -> vector<i1> {
  %v = vector.create_mask %num_elems : vector<i1>
  return %v: vector<i1>
}

// CHECK-LABEL: func @create_mask_0d
// CHECK-SAME: %[[NUM_ELEMS:.*]]: index
// CHECK:  %[[INDICES:.*]] = arith.constant dense<0> : vector<i32>
// CHECK:  %[[NUM_ELEMS_i32:.*]] = arith.index_cast %[[NUM_ELEMS]] : index to i32
// CHECK:  %[[BOUNDS:.*]] = llvm.insertelement %[[NUM_ELEMS_i32]]
// CHECK:  %[[BOUNDS_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOUNDS]] : vector<1xi32> to vector<i32>
// CHECK:  %[[RESULT:.*]] = arith.cmpi sgt, %[[BOUNDS_CAST]], %[[INDICES]] : vector<i32>
// CHECK:  return %[[RESULT]] : vector<i1>

// -----

func.func @create_mask_1d(%num_elems : index) -> vector<4xi1> {
  %v = vector.create_mask %num_elems : vector<4xi1>
  return %v: vector<4xi1>
}

// CHECK-LABEL: func @create_mask_1d
// CHECK-SAME: %[[NUM_ELEMS:.*]]: index
// CHECK:  %[[INDICES:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
// CHECK:  %[[NUM_ELEMS_i32:.*]] = arith.index_cast %[[NUM_ELEMS]] : index to i32
// CHECK:  %[[BOUNDS_INSERT:.*]] = llvm.insertelement %[[NUM_ELEMS_i32]]
// CHECK:  %[[BOUNDS:.*]] = llvm.shufflevector %[[BOUNDS_INSERT]]
// CHECK:  %[[RESULT:.*]] = arith.cmpi sgt, %[[BOUNDS]], %[[INDICES]] : vector<4xi32>
// CHECK:  return %[[RESULT]] : vector<4xi1>

// -----

func.func @create_mask_1d_scalable(%num_elems : index) -> vector<[4]xi1> {
  %v = vector.create_mask %num_elems : vector<[4]xi1>
  return %v: vector<[4]xi1>
}

// CHECK-LABEL: func @create_mask_1d_scalable
// CHECK-SAME: %[[NUM_ELEMS:.*]]: index
// CHECK:  %[[INDICES:.*]] = llvm.intr.stepvector : vector<[4]xi32>
// CHECK:  %[[NUM_ELEMS_i32:.*]] = arith.index_cast %[[NUM_ELEMS]] : index to i32
// CHECK:  %[[BOUNDS_INSERT:.*]] = llvm.insertelement %[[NUM_ELEMS_i32]], {{.*}} : vector<[4]xi32>
// CHECK:  %[[BOUNDS:.*]] = llvm.shufflevector %[[BOUNDS_INSERT]], {{.*}} : vector<[4]xi32>
// CHECK:  %[[RESULT:.*]] = arith.cmpi slt, %[[INDICES]], %[[BOUNDS]] : vector<[4]xi32>
// CHECK: return %[[RESULT]] : vector<[4]xi1>

// -----

//===----------------------------------------------------------------------===//
// vector.gather
//
// NOTE: vector.constant_mask won't lower with
//  * --convert-to-llvm="filter-dialects=vector",
// hence testing here.
//===----------------------------------------------------------------------===//


func.func @gather_with_mask(%arg0: memref<?xf32>, %arg1: vector<2x3xi32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.constant_mask [2, 2] : vector<2x3xi1>
  %2 = vector.gather %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x3xi32>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %2 : vector<2x3xf32>
}

// CHECK-LABEL: func @gather_with_mask
// CHECK: %[[G0:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: %[[G1:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>

// -----

func.func @gather_with_mask_scalable(%arg0: memref<?xf32>, %arg1: vector<2x[3]xi32>, %arg2: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = arith.constant 0: index
  // vector.constant_mask only supports 'none set' or 'all set' scalable
  // dimensions, hence [2, 3] rather than [2, 2] as in the example for fixed
  // width vectors above.
  %1 = vector.constant_mask [2, 3] : vector<2x[3]xi1>
  %2 = vector.gather %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x[3]xi32>, vector<2x[3]xi1>, vector<2x[3]xf32> into vector<2x[3]xf32>
  return %2 : vector<2x[3]xf32>
}

// CHECK-LABEL: func @gather_with_mask_scalable
// CHECK: %[[G0:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<[3]x!llvm.ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: %[[G1:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<[3]x!llvm.ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>


// -----

func.func @gather_with_zero_mask(%arg0: memref<?xf32>, %arg1: vector<2x3xi32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.constant_mask [0, 0] : vector<2x3xi1>
  %2 = vector.gather %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x3xi32>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %2 : vector<2x3xf32>
}

// CHECK-LABEL: func @gather_with_zero_mask
// CHECK-SAME:    (%{{.*}}: memref<?xf32>, %{{.*}}: vector<2x3xi32>, %[[S:.*]]: vector<2x3xf32>)
// CHECK-NOT:   %{{.*}} = llvm.intr.masked.gather
// CHECK:       return %[[S]] : vector<2x3xf32>

// -----

func.func @gather_with_zero_mask_scalable(%arg0: memref<?xf32>, %arg1: vector<2x[3]xi32>, %arg2: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = arith.constant 0: index
  %1 = vector.constant_mask [0, 0] : vector<2x[3]xi1>
  %2 = vector.gather %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x[3]xi32>, vector<2x[3]xi1>, vector<2x[3]xf32> into vector<2x[3]xf32>
  return %2 : vector<2x[3]xf32>
}

// CHECK-LABEL: func @gather_with_zero_mask_scalable
// CHECK-SAME:    (%{{.*}}: memref<?xf32>, %{{.*}}: vector<2x[3]xi32>, %[[S:.*]]: vector<2x[3]xf32>)
// CHECK-NOT:   %{{.*}} = llvm.intr.masked.gather
// CHECK:       return %[[S]] : vector<2x[3]xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.scatter
//===----------------------------------------------------------------------===//

// Multi-Dimensional scatters are not supported yet. Check that we do not lower
// them.

func.func @scatter_with_mask(%arg0: memref<?xf32>, %arg1: vector<2x3xi32>, %arg2: vector<2x3xf32>) {
  %0 = arith.constant 0: index
  %1 = vector.constant_mask [2, 2] : vector<2x3xi1>
  vector.scatter %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x3xi32>, vector<2x3xi1>, vector<2x3xf32>
  return
}

// CHECK-LABEL: func @scatter_with_mask
// CHECK: vector.scatter

// -----

func.func @scatter_with_mask_scalable(%arg0: memref<?xf32>, %arg1: vector<2x[3]xi32>, %arg2: vector<2x[3]xf32>) {
  %0 = arith.constant 0: index
  // vector.constant_mask only supports 'none set' or 'all set' scalable
  // dimensions, hence [2, 3] rather than [2, 2] as in the example for fixed
  // width vectors above.
  %1 = vector.constant_mask [2, 3] : vector<2x[3]xi1>
  vector.scatter %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x[3]xi32>, vector<2x[3]xi1>, vector<2x[3]xf32>
  return
}

// CHECK-LABEL: func @scatter_with_mask_scalable
// CHECK: vector.scatter

// -----

//===----------------------------------------------------------------------===//
// vector.interleave
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @interleave_2d
//  CHECK-SAME:     %[[LHS:.*]]: vector<2x3xi8>, %[[RHS:.*]]: vector<2x3xi8>)
func.func @interleave_2d(%a: vector<2x3xi8>, %b: vector<2x3xi8>) -> vector<2x6xi8> {
  // CHECK: llvm.shufflevector
  // CHECK-NOT: vector.interleave {{.*}} : vector<2x3xi8>
  %0 = vector.interleave %a, %b : vector<2x3xi8> -> vector<2x6xi8>
  return %0 : vector<2x6xi8>
}

// -----

// CHECK-LABEL: @interleave_2d_scalable
//  CHECK-SAME:     %[[LHS:.*]]: vector<2x[8]xi16>, %[[RHS:.*]]: vector<2x[8]xi16>)
func.func @interleave_2d_scalable(%a: vector<2x[8]xi16>, %b: vector<2x[8]xi16>) -> vector<2x[16]xi16> {
  // CHECK: llvm.intr.vector.interleave2
  // CHECK-NOT: vector.interleave {{.*}} : vector<2x[8]xi16>
  %0 = vector.interleave %a, %b : vector<2x[8]xi16> -> vector<2x[16]xi16>
  return %0 : vector<2x[16]xi16>
}

// -----

//===----------------------------------------------------------------------===//
// vector.deinterleave
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @deinterleave_2d
// CHECK-SAME: %[[ARG:.*]]: vector<2x8xf32>) -> (vector<2x4xf32>, vector<2x4xf32>)
func.func @deinterleave_2d(%arg: vector<2x8xf32>) -> (vector<2x4xf32>, vector<2x4xf32>) {
  // CHECK: llvm.shufflevector
  // CHECK-NOT: vector.deinterleave %{{.*}} : vector<2x8xf32>
  %0, %1 = vector.deinterleave %arg : vector<2x8xf32> -> vector<2x4xf32>
  return %0, %1 : vector<2x4xf32>, vector<2x4xf32>
}

// -----

func.func @deinterleave_2d_scalable(%arg: vector<2x[8]xf32>) -> (vector<2x[4]xf32>, vector<2x[4]xf32>) {
    // CHECK: llvm.intr.vector.deinterleave2
    // CHECK-NOT: vector.deinterleave %{{.*}} : vector<2x[8]xf32>
    %0, %1 = vector.deinterleave %arg : vector<2x[8]xf32> -> vector<2x[4]xf32>
    return %0, %1 : vector<2x[4]xf32>, vector<2x[4]xf32>
}


// -----

//===----------------------------------------------------------------------===//
// vector.step
//===----------------------------------------------------------------------===//

// TODO: Investigate why this wouldn't lower with --convert-to-llvm="filter-dialects=vector"

// CHECK-LABEL: @step
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK: return %[[CST]] : vector<4xindex>
func.func @step() -> vector<4xindex> {
  %0 = vector.step : vector<4xindex>
  return %0 : vector<4xindex>
}


// -----

//===----------------------------------------------------------------------===//
// vector.from_elements
//===----------------------------------------------------------------------===//

// NOTE: We unroll multi-dimensional from_elements ops with pattern `UnrollFromElements`
// and then convert the 1-D from_elements ops to llvm.

// CHECK-LABEL: func @from_elements_3d
//  CHECK-SAME:  %[[ARG_0:.*]]: f32, %[[ARG_1:.*]]: f32, %[[ARG_2:.*]]: f32, %[[ARG_3:.*]]: f32)
//       CHECK:  %[[UNDEF_RES:.*]] = ub.poison : vector<2x1x2xf32>
//       CHECK:  %[[UNDEF_RES_LLVM:.*]] = builtin.unrealized_conversion_cast %[[UNDEF_RES]] : vector<2x1x2xf32> to !llvm.array<2 x array<1 x vector<2xf32>>>
//       CHECK:  %[[UNDEF_VEC_RANK_2:.*]] = ub.poison : vector<1x2xf32>
//       CHECK:  %[[UNDEF_VEC_RANK_2_LLVM:.*]] = builtin.unrealized_conversion_cast %[[UNDEF_VEC_RANK_2]] : vector<1x2xf32> to !llvm.array<1 x vector<2xf32>>
//       CHECK:  %[[UNDEF_VEC0:.*]] = llvm.mlir.poison : vector<2xf32>
//       CHECK:  %[[C0_0:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:  %[[VEC0_0:.*]] = llvm.insertelement %[[ARG_0]], %[[UNDEF_VEC0]][%[[C0_0]] : i64] : vector<2xf32>
//       CHECK:  %[[C1_0:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:  %[[VEC0_1:.*]] = llvm.insertelement %[[ARG_1]], %[[VEC0_0]][%[[C1_0]] : i64] : vector<2xf32>
//       CHECK:  %[[RES_RANK_2_0:.*]] = llvm.insertvalue %[[VEC0_1]], %[[UNDEF_VEC_RANK_2_LLVM]][0] : !llvm.array<1 x vector<2xf32>>
//       CHECK:  %[[RES_0:.*]] = llvm.insertvalue %[[RES_RANK_2_0]], %[[UNDEF_RES_LLVM]][0] : !llvm.array<2 x array<1 x vector<2xf32>>>
//       CHECK:  %[[UNDEF_VEC1:.*]] = llvm.mlir.poison : vector<2xf32>
//       CHECK:  %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:  %[[VEC1_0:.*]] = llvm.insertelement %[[ARG_2]], %[[UNDEF_VEC1]][%[[C0_1]] : i64] : vector<2xf32>
//       CHECK:  %[[C1_1:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:  %[[VEC1_1:.*]] = llvm.insertelement %[[ARG_3]], %[[VEC1_0]][%[[C1_1]] : i64] : vector<2xf32>
//       CHECK:  %[[RES_RANK_2_1:.*]] = llvm.insertvalue %[[VEC1_1]], %[[UNDEF_VEC_RANK_2_LLVM]][0] : !llvm.array<1 x vector<2xf32>>
//       CHECK:  %[[RES_1:.*]] = llvm.insertvalue %[[RES_RANK_2_1]], %[[RES_0]][1] : !llvm.array<2 x array<1 x vector<2xf32>>>
//       CHECK:  %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[RES_1]] : !llvm.array<2 x array<1 x vector<2xf32>>> to vector<2x1x2xf32>
//       CHECK:  return %[[CAST]]
func.func @from_elements_3d(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> vector<2x1x2xf32> {
  %0 = vector.from_elements %arg0, %arg1, %arg2, %arg3 : vector<2x1x2xf32>
  return %0 : vector<2x1x2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// vector.to_elements
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @to_elements_1d(
// CHECK-SAME:    %[[ARG0:.+]]: vector<2xf32>
// CHECK:         %[[C0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[V0:.+]] = llvm.extractelement %[[ARG0]][%[[C0]] : i64] : vector<2xf32>
// CHECK:         %[[C1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[V1:.+]] = llvm.extractelement %[[ARG0]][%[[C1]] : i64] : vector<2xf32>
// CHECK:         return %[[V0]], %[[V1]]
func.func @to_elements_1d(%arg0: vector<2xf32>) -> (f32, f32) {
  %0:2 = vector.to_elements %arg0 : vector<2xf32>
  return %0#0, %0#1 : f32, f32
}

// -----

// NOTE: We unroll multi-dimensional to_elements ops with pattern
// `UnrollToElements` and then convert the 1-D to_elements ops to llvm.

// CHECK-LABEL: func @to_elements_2d(
// CHECK-SAME:    %[[ARG0:.+]]: vector<2x2xf32>
// CHECK:         %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<2x2xf32> to !llvm.array<2 x vector<2xf32>>
// CHECK:         %[[V0:.+]] = llvm.extractvalue %[[CAST]][0] : !llvm.array<2 x vector<2xf32>>
// CHECK:         %[[V1:.+]] = llvm.extractvalue %[[CAST]][1] : !llvm.array<2 x vector<2xf32>>
// CHECK:         %[[C0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[R0:.+]] = llvm.extractelement %[[V0]][%[[C0]] : i64] : vector<2xf32>
// CHECK:         %[[C1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[R1:.+]] = llvm.extractelement %[[V0]][%[[C1]] : i64] : vector<2xf32>
// CHECK:         %[[C0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[R2:.+]] = llvm.extractelement %[[V1]][%[[C0]] : i64] : vector<2xf32>
// CHECK:         %[[C1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[R3:.+]] = llvm.extractelement %[[V1]][%[[C1]] : i64] : vector<2xf32>
// CHECK:         return %[[R0]], %[[R1]], %[[R2]], %[[R3]]
func.func @to_elements_2d(%arg0: vector<2x2xf32>) -> (f32, f32, f32, f32) {
  %0:4 = vector.to_elements %arg0 : vector<2x2xf32>
  return %0#0, %0#1, %0#2, %0#3 : f32, f32, f32, f32
}
