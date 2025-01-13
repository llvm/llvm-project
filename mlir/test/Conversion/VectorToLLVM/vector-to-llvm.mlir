// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// vector.bitcast
//===----------------------------------------------------------------------===//

func.func @bitcast_f32_to_i32_vector_0d(%arg0: vector<f32>) -> vector<i32> {
  %0 = vector.bitcast %arg0 : vector<f32> to vector<i32>
  return %0 : vector<i32>
}

// CHECK-LABEL: @bitcast_f32_to_i32_vector_0d
// CHECK-SAME:  %[[ARG_0:.*]]: vector<f32>
// CHECK:       %[[VEC_F32_1D:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : vector<f32> to vector<1xf32>
// CHECK:       %[[VEC_I32_1D:.*]] = llvm.bitcast %[[VEC_F32_1D]] : vector<1xf32> to vector<1xi32>
// CHECK:       %[[VEC_I32_0D:.*]] = builtin.unrealized_conversion_cast %[[VEC_I32_1D]] : vector<1xi32> to vector<i32>
// CHECK:       return %[[VEC_I32_0D]] : vector<i32>

// -----

func.func @bitcast_f32_to_i32_vector(%arg0: vector<16xf32>) -> vector<16xi32> {
  %0 = vector.bitcast %arg0 : vector<16xf32> to vector<16xi32>
  return %0 : vector<16xi32>
}


// CHECK-LABEL: @bitcast_f32_to_i32_vector
// CHECK-SAME:  %[[ARG_0:.*]]: vector<16xf32>
// CHECK:       llvm.bitcast %[[ARG_0]] : vector<16xf32> to vector<16xi32>

// -----

func.func @bitcast_f32_to_i32_vector_scalable(%arg0: vector<[16]xf32>) -> vector<[16]xi32> {
  %0 = vector.bitcast %arg0 : vector<[16]xf32> to vector<[16]xi32>
  return %0 : vector<[16]xi32>
}

// CHECK-LABEL: @bitcast_f32_to_i32_vector_scalable
// CHECK-SAME:  %[[ARG_0:.*]]: vector<[16]xf32>
// CHECK:       llvm.bitcast %[[ARG_0]] : vector<[16]xf32> to vector<[16]xi32>

// -----

func.func @bitcast_i8_to_f32_vector(%arg0: vector<64xi8>) -> vector<16xf32> {
  %0 = vector.bitcast %arg0 : vector<64xi8> to vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @bitcast_i8_to_f32_vector
// CHECK-SAME:  %[[ARG_0:.*]]: vector<64xi8>
// CHECK:       llvm.bitcast %[[ARG_0]] : vector<64xi8> to vector<16xf32>

// -----

func.func @bitcast_i8_to_f32_vector_scalable(%arg0: vector<[64]xi8>) -> vector<[16]xf32> {
  %0 = vector.bitcast %arg0 : vector<[64]xi8> to vector<[16]xf32>
  return %0 : vector<[16]xf32>
}

// CHECK-LABEL: @bitcast_i8_to_f32_vector_scalable
// CHECK-SAME:  %[[ARG_0:.*]]: vector<[64]xi8>
// CHECK:       llvm.bitcast %[[ARG_0]] : vector<[64]xi8> to vector<[16]xf32>

// -----

func.func @bitcast_index_to_i8_vector(%arg0: vector<16xindex>) -> vector<128xi8> {
  %0 = vector.bitcast %arg0 : vector<16xindex> to vector<128xi8>
  return %0 : vector<128xi8>
}

// CHECK-LABEL: @bitcast_index_to_i8_vector
// CHECK-SAME:  %[[ARG_0:.*]]: vector<16xindex>
// CHECK:       %[[T0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : vector<16xindex> to vector<16xi64>
// CHECK:       llvm.bitcast %[[T0]] : vector<16xi64> to vector<128xi8>

// -----

func.func @bitcast_index_to_i8_vector_scalable(%arg0: vector<[16]xindex>) -> vector<[128]xi8> {
  %0 = vector.bitcast %arg0 : vector<[16]xindex> to vector<[128]xi8>
  return %0 : vector<[128]xi8>
}

// CHECK-LABEL: @bitcast_index_to_i8_vector_scalable
// CHECK-SAME:  %[[ARG_0:.*]]: vector<[16]xindex>
// CHECK:       %[[T0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : vector<[16]xindex> to vector<[16]xi64>
// CHECK:       llvm.bitcast %[[T0]] : vector<[16]xi64> to vector<[128]xi8>

// -----

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

func.func @broadcast_vec0d_from_vec0d(%arg0: vector<f32>) -> vector<f32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<f32>
  return %0 : vector<f32>
}
// CHECK-LABEL: @broadcast_vec0d_from_vec0d(
// CHECK-SAME:  %[[A:.*]]: vector<f32>)
// CHECK:       return %[[A]] : vector<f32>

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

func.func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: @broadcast_vec1d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: vector<2xf32>)
// CHECK:       return %[[A]] : vector<2xf32>

// -----

func.func @broadcast_vec1d_from_vec1d_scalable(%arg0: vector<[2]xf32>) -> vector<[2]xf32> {
  %0 = vector.broadcast %arg0 : vector<[2]xf32> to vector<[2]xf32>
  return %0 : vector<[2]xf32>
}
// CHECK-LABEL: @broadcast_vec1d_from_vec1d_scalable(
// CHECK-SAME:  %[[A:.*]]: vector<[2]xf32>)
// CHECK:       return %[[A]] : vector<[2]xf32>

// -----

func.func @broadcast_vec2d_from_vec0d(%arg0: vector<f32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: @broadcast_vec2d_from_vec0d(
// CHECK-SAME:  %[[A:.*]]: vector<f32>)
//       CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<f32> to vector<1xf32>
//       CHECK: %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
//       CHECK: %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
//       CHECK: %[[T4:.*]] = llvm.mlir.constant(0 : index) : i64
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
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
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
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<3x[2]xf32>
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
// CHECK:       %[[T0:.*]] = arith.constant dense<0> : vector<3x2xindex>
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
// CHECK:       %[[T0:.*]] = arith.constant dense<0> : vector<3x[2]xindex>
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
// CHECK-DAG:   %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK-DAG:   %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x2xf32> to !llvm.array<3 x vector<2xf32>>
// CHECK-DAG:   %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
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
// CHECK-DAG:   %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<3x[2]xf32>
// CHECK-DAG:   %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<3x[2]xf32> to !llvm.array<3 x vector<[2]xf32>>
// CHECK-DAG:   %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x[2]xf32>
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
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
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
// CHECK:       %[[T0:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x[2]xf32>
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
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<3x4xf32>
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
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<3x[4]xf32>
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
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3xf32>
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
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:       %[[T9:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x2xf32> to !llvm.array<4 x array<3 x vector<2xf32>>>
// CHECK:       %[[T2:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
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
// CHECK:       %[[T1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x[2]xf32>
// CHECK:       %[[T9:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<4x3x[2]xf32> to !llvm.array<4 x array<3 x vector<[2]xf32>>>
// CHECK:       %[[T2:.*]] = arith.constant dense<0.000000e+00> : vector<3x[2]xf32>
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
// vector.shuffle
//===----------------------------------------------------------------------===//

func.func @shuffle_0D_direct(%arg0: vector<f32>) -> vector<3xf32> {
  %1 = vector.shuffle %arg0, %arg0 [0, 1, 0] : vector<f32>, vector<f32>
  return %1 : vector<3xf32>
}
// CHECK-LABEL: @shuffle_0D_direct(
//  CHECK-SAME:     %[[A:.*]]: vector<f32>
//       CHECK:   %[[c:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<f32> to vector<1xf32>
//       CHECK:   %[[s:.*]] = llvm.shufflevector %[[c]], %[[c]] [0, 1, 0] : vector<1xf32>
//       CHECK:   return %[[s]] : vector<3xf32>

// -----

func.func @shuffle_1D_direct(%arg0: vector<2xf32>, %arg1: vector<2xf32>) -> vector<2xf32> {
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xf32>, vector<2xf32>
  return %1 : vector<2xf32>
}
// CHECK-LABEL: @shuffle_1D_direct(
// CHECK-SAME: %[[A:.*]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*]]: vector<2xf32>)
//       CHECK:   return %[[A:.*]]: vector<2xf32>

// -----

func.func @shuffle_1D_index_direct(%arg0: vector<2xindex>, %arg1: vector<2xindex>) -> vector<2xindex> {
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xindex>, vector<2xindex>
  return %1 : vector<2xindex>
}
// CHECK-LABEL: @shuffle_1D_index_direct(
// CHECK-SAME: %[[A:.*]]: vector<2xindex>,
// CHECK-SAME: %[[B:.*]]: vector<2xindex>)
//       CHECK:   return  %[[A:.*]]: vector<2xindex>

// -----

func.func @shuffle_1D(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<5xf32> {
  %1 = vector.shuffle %arg0, %arg1 [4, 3, 2, 1, 0] : vector<2xf32>, vector<3xf32>
  return %1 : vector<5xf32>
}
// CHECK-LABEL: @shuffle_1D(
// CHECK-SAME: %[[A:.*]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*]]: vector<3xf32>)
//       CHECK:   %[[U0:.*]] = llvm.mlir.undef : vector<5xf32>
//       CHECK:   %[[C2:.*]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   %[[E1:.*]] = llvm.extractelement %[[B]][%[[C2]] : i64] : vector<3xf32>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[I1:.*]] = llvm.insertelement %[[E1]], %[[U0]][%[[C0]] : i64] : vector<5xf32>
//       CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   %[[E2:.*]] = llvm.extractelement %[[B]][%[[C1]] : i64] : vector<3xf32>
//       CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   %[[I2:.*]] = llvm.insertelement %[[E2]], %[[I1]][%[[C1]] : i64] : vector<5xf32>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[E3:.*]] = llvm.extractelement %[[B]][%[[C0]] : i64] : vector<3xf32>
//       CHECK:   %[[C2:.*]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   %[[I3:.*]] = llvm.insertelement %[[E3]], %[[I2]][%[[C2]] : i64] : vector<5xf32>
//       CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   %[[E4:.*]] = llvm.extractelement %[[A]][%[[C1]] : i64] : vector<2xf32>
//       CHECK:   %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
//       CHECK:   %[[I4:.*]] = llvm.insertelement %[[E4]], %[[I3]][%[[C3]] : i64] : vector<5xf32>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[E5:.*]] = llvm.extractelement %[[A]][%[[C0]] : i64] : vector<2xf32>
//       CHECK:   %[[C4:.*]] = llvm.mlir.constant(4 : index) : i64
//       CHECK:   %[[I5:.*]] = llvm.insertelement %[[E5]], %[[I4]][%[[C4]] : i64] : vector<5xf32>
//       CHECK:   return %[[I5]] : vector<5xf32>

// -----

func.func @shuffle_2D(%a: vector<1x4xf32>, %b: vector<2x4xf32>) -> vector<3x4xf32> {
  %1 = vector.shuffle %a, %b[1, 0, 2] : vector<1x4xf32>, vector<2x4xf32>
  return %1 : vector<3x4xf32>
}
// CHECK-LABEL: @shuffle_2D(
// CHECK-SAME: %[[A:.*]]: vector<1x4xf32>,
// CHECK-SAME: %[[B:.*]]: vector<2x4xf32>)
//       CHECK-DAG:   %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<1x4xf32> to !llvm.array<1 x vector<4xf32>>
//       CHECK-DAG:   %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<2x4xf32> to !llvm.array<2 x vector<4xf32>>
//       CHECK:   %[[U0:.*]] = llvm.mlir.undef : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[E1:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.array<2 x vector<4xf32>>
//       CHECK:   %[[I1:.*]] = llvm.insertvalue %[[E1]], %[[U0]][0] : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[E2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.array<1 x vector<4xf32>>
//       CHECK:   %[[I2:.*]] = llvm.insertvalue %[[E2]], %[[I1]][1] : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[E3:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.array<2 x vector<4xf32>>
//       CHECK:   %[[I3:.*]] = llvm.insertvalue %[[E3]], %[[I2]][2] : !llvm.array<3 x vector<4xf32>>
//       CHECK:   %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[I3]] : !llvm.array<3 x vector<4xf32>> to vector<3x4xf32>
//       CHECK:   return %[[VAL_3]] : vector<3x4xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.extractelement
//===----------------------------------------------------------------------===//

func.func @extractelement_from_vec_0d_f32(%arg0: vector<f32>) -> f32 {
  %1 = vector.extractelement %arg0[] : vector<f32>
  return %1 : f32
}
// CHECK-LABEL: @extractelement_from_vec_0d_f32
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   llvm.extractelement %{{.*}}[%[[C0]] : {{.*}}] : vector<1xf32>

// -----

func.func @extractelement_from_vec_1d_f32_idx_as_i32(%arg0: vector<16xf32>) -> f32 {
  %0 = arith.constant 15 : i32
  %1 = vector.extractelement %arg0[%0 : i32]: vector<16xf32>
  return %1 : f32
}
// CHECK-LABEL: @extractelement_from_vec_1d_f32_idx_as_i32(
//  CHECK-SAME:   %[[A:.*]]: vector<16xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 15 : i32
//       CHECK:   %[[X:.*]] = llvm.extractelement %[[A]][%[[C]] : i32] : vector<16xf32>
//       CHECK:   return %[[X]] : f32

// -----

func.func @extractelement_from_vec_1d_f32_idx_as_i32_scalable(%arg0: vector<[16]xf32>) -> f32 {
  %0 = arith.constant 15 : i32
  %1 = vector.extractelement %arg0[%0 : i32]: vector<[16]xf32>
  return %1 : f32
}
// CHECK-LABEL: @extractelement_from_vec_1d_f32_idx_as_i32_scalable(
//  CHECK-SAME:   %[[A:.*]]: vector<[16]xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 15 : i32
//       CHECK:   %[[X:.*]] = llvm.extractelement %[[A]][%[[C]] : i32] : vector<[16]xf32>
//       CHECK:   return %[[X]] : f32

// -----
func.func @extractelement_from_vec_1d_f32_idx_as_index(%arg0: vector<16xf32>) -> f32 {
  %0 = arith.constant 15 : index
  %1 = vector.extractelement %arg0[%0 : index]: vector<16xf32>
  return %1 : f32
}
// CHECK-LABEL: @extractelement_from_vec_1d_f32_idx_as_index(
//  CHECK-SAME:   %[[A:.*]]: vector<16xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 15 : index
//       CHECK:   %[[I:.*]] = builtin.unrealized_conversion_cast %[[C]] : index to i64
//       CHECK:   %[[X:.*]] = llvm.extractelement %[[A]][%[[I]] : i64] : vector<16xf32>
//       CHECK:   return %[[X]] : f32

// -----

func.func @extractelement_from_vec_1d_f32_idx_as_index_scalable(%arg0: vector<[16]xf32>) -> f32 {
  %0 = arith.constant 15 : index
  %1 = vector.extractelement %arg0[%0 : index]: vector<[16]xf32>
  return %1 : f32
}
// CHECK-LABEL: @extractelement_from_vec_1d_f32_idx_as_index_scalable(
//  CHECK-SAME:   %[[A:.*]]: vector<[16]xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 15 : index
//       CHECK:   %[[I:.*]] = builtin.unrealized_conversion_cast %[[C]] : index to i64
//       CHECK:   %[[X:.*]] = llvm.extractelement %[[A]][%[[I]] : i64] : vector<[16]xf32>
//       CHECK:   return %[[X]] : f32

// -----

//===----------------------------------------------------------------------===//
// vector.extract
//===----------------------------------------------------------------------===//

func.func @extract_scalar_from_vec_1d_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.extract %arg0[15]: f32 from vector<16xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_f32
//       CHECK:   llvm.mlir.constant(15 : i64) : i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : i64] : vector<16xf32>
//       CHECK:   return {{.*}} : f32

// -----

func.func @extract_scalar_from_vec_1d_f32_scalable(%arg0: vector<[16]xf32>) -> f32 {
  %0 = vector.extract %arg0[15]: f32 from vector<[16]xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_f32_scalable
//       CHECK:   llvm.mlir.constant(15 : i64) : i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : i64] : vector<[16]xf32>
//       CHECK:   return {{.*}} : f32

// -----

func.func @extract_vec_1e_from_vec_1d_f32(%arg0: vector<16xf32>) -> vector<1xf32> {
  %0 = vector.extract %arg0[15]: vector<1xf32> from vector<16xf32>
  return %0 : vector<1xf32>
}
// CHECK-LABEL: @extract_vec_1e_from_vec_1d_f32(
//  CHECK-SAME:   %[[A:.*]]: vector<16xf32>)
//       CHECK:   %[[T0:.*]] = llvm.mlir.constant(15 : i64) : i64
//       CHECK:   %[[T1:.*]] = llvm.extractelement %[[A]][%[[T0]] : i64] : vector<16xf32>
//       CHECK:   %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : f32 to vector<1xf32>
//       CHECK:   return %[[T2]] : vector<1xf32>

// -----

func.func @extract_vec_1e_from_vec_1d_f32_scalable(%arg0: vector<[16]xf32>) -> vector<1xf32> {
  %0 = vector.extract %arg0[15]: vector<1xf32> from vector<[16]xf32>
  return %0 : vector<1xf32>
}
// CHECK-LABEL: @extract_vec_1e_from_vec_1d_f32_scalable(
//  CHECK-SAME:   %[[A:.*]]: vector<[16]xf32>)
//       CHECK:   %[[T0:.*]] = llvm.mlir.constant(15 : i64) : i64
//       CHECK:   %[[T1:.*]] = llvm.extractelement %[[A]][%[[T0]] : i64] : vector<[16]xf32>
//       CHECK:   %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : f32 to vector<1xf32>
//       CHECK:   return %[[T2]] : vector<1xf32>

// -----

func.func @extract_scalar_from_vec_1d_index(%arg0: vector<16xindex>) -> index {
  %0 = vector.extract %arg0[15]: index from vector<16xindex>
  return %0 : index
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_index(
//  CHECK-SAME:   %[[A:.*]]: vector<16xindex>)
//       CHECK:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<16xindex> to vector<16xi64>
//       CHECK:   %[[T1:.*]] = llvm.mlir.constant(15 : i64) : i64
//       CHECK:   %[[T2:.*]] = llvm.extractelement %[[T0]][%[[T1]] : i64] : vector<16xi64>
//       CHECK:   %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : i64 to index
//       CHECK:   return %[[T3]] : index

// -----

func.func @extract_scalar_from_vec_1d_index_scalable(%arg0: vector<[16]xindex>) -> index {
  %0 = vector.extract %arg0[15]: index from vector<[16]xindex>
  return %0 : index
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_index_scalable(
//  CHECK-SAME:   %[[A:.*]]: vector<[16]xindex>)
//       CHECK:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<[16]xindex> to vector<[16]xi64>
//       CHECK:   %[[T1:.*]] = llvm.mlir.constant(15 : i64) : i64
//       CHECK:   %[[T2:.*]] = llvm.extractelement %[[T0]][%[[T1]] : i64] : vector<[16]xi64>
//       CHECK:   %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : i64 to index
//       CHECK:   return %[[T3]] : index

// -----

func.func @extract_vec_2d_from_vec_3d_f32(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extract %arg0[0]: vector<3x16xf32> from vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: @extract_vec_2d_from_vec_3d_f32
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm.array<4 x array<3 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<3x16xf32>

// -----

func.func @extract_vec_2d_from_vec_3d_f32_scalable(%arg0: vector<4x3x[16]xf32>) -> vector<3x[16]xf32> {
  %0 = vector.extract %arg0[0]: vector<3x[16]xf32> from vector<4x3x[16]xf32>
  return %0 : vector<3x[16]xf32>
}
// CHECK-LABEL: @extract_vec_2d_from_vec_3d_f32_scalable
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm.array<4 x array<3 x vector<[16]xf32>>>
//       CHECK:   return {{.*}} : vector<3x[16]xf32>

// -----

func.func @extract_vec_1d_from_vec_3d_f32(%arg0: vector<4x3x16xf32>) -> vector<16xf32> {
  %0 = vector.extract %arg0[0, 0]: vector<16xf32> from vector<4x3x16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: @extract_vec_1d_from_vec_3d_f32
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm.array<4 x array<3 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<16xf32>

// -----

func.func @extract_vec_1d_from_vec_3d_f32_scalable(%arg0: vector<4x3x[16]xf32>) -> vector<[16]xf32> {
  %0 = vector.extract %arg0[0, 0]: vector<[16]xf32> from vector<4x3x[16]xf32>
  return %0 : vector<[16]xf32>
}
// CHECK-LABEL: @extract_vec_1d_from_vec_3d_f32_scalable
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm.array<4 x array<3 x vector<[16]xf32>>>
//       CHECK:   return {{.*}} : vector<[16]xf32>

// -----

func.func @extract_scalar_from_vec_3d_f32(%arg0: vector<4x3x16xf32>) -> f32 {
  %0 = vector.extract %arg0[0, 0, 0]: f32 from vector<4x3x16xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_3d_f32
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm.array<4 x array<3 x vector<16xf32>>>
//       CHECK:   llvm.mlir.constant(0 : i64) : i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : i64] : vector<16xf32>
//       CHECK:   return {{.*}} : f32

// -----

func.func @extract_scalar_from_vec_3d_f32_scalable(%arg0: vector<4x3x[16]xf32>) -> f32 {
  %0 = vector.extract %arg0[0, 0, 0]: f32 from vector<4x3x[16]xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_3d_f32_scalable
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm.array<4 x array<3 x vector<[16]xf32>>>
//       CHECK:   llvm.mlir.constant(0 : i64) : i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : i64] : vector<[16]xf32>
//       CHECK:   return {{.*}} : f32

// -----

func.func @extract_scalar_from_vec_1d_f32_dynamic_idx(%arg0: vector<16xf32>, %arg1: index) -> f32 {
  %0 = vector.extract %arg0[%arg1]: f32 from vector<16xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_f32_dynamic_idx
//  CHECK-SAME:   %[[VEC:.+]]: vector<16xf32>, %[[INDEX:.+]]: index
//       CHECK:   %[[UC:.+]] = builtin.unrealized_conversion_cast %[[INDEX]] : index to i64
//       CHECK:   llvm.extractelement %[[VEC]][%[[UC]] : i64] : vector<16xf32>

// -----

func.func @extract_scalar_from_vec_1d_f32_dynamic_idx_scalable(%arg0: vector<[16]xf32>, %arg1: index) -> f32 {
  %0 = vector.extract %arg0[%arg1]: f32 from vector<[16]xf32>
  return %0 : f32
}
// CHECK-LABEL: @extract_scalar_from_vec_1d_f32_dynamic_idx_scalable
//  CHECK-SAME:   %[[VEC:.+]]: vector<[16]xf32>, %[[INDEX:.+]]: index
//       CHECK:   %[[UC:.+]] = builtin.unrealized_conversion_cast %[[INDEX]] : index to i64
//       CHECK:   llvm.extractelement %[[VEC]][%[[UC]] : i64] : vector<[16]xf32>

// -----

func.func @extract_scalar_from_vec_2d_f32_inner_dynamic_idx(%arg0: vector<1x16xf32>, %arg1: index) -> f32 {
  %0 = vector.extract %arg0[0, %arg1]: f32 from vector<1x16xf32>
  return %0 : f32
}

// Lowering supports extracting from multi-dim vectors with dynamic indices
// provided that only the trailing index is dynamic.

// CHECK-LABEL: @extract_scalar_from_vec_2d_f32_inner_dynamic_idx(
//       CHECK:   llvm.extractvalue
//       CHECK:   llvm.extractelement

func.func @extract_scalar_from_vec_2d_f32_inner_dynamic_idx_scalable(%arg0: vector<1x[16]xf32>, %arg1: index) -> f32 {
  %0 = vector.extract %arg0[0, %arg1]: f32 from vector<1x[16]xf32>
  return %0 : f32
}

// Lowering supports extracting from multi-dim vectors with dynamic indices
// provided that only the trailing index is dynamic.

// CHECK-LABEL: @extract_scalar_from_vec_2d_f32_inner_dynamic_idx_scalable(
//       CHECK:   llvm.extractvalue
//       CHECK:   llvm.extractelement

// -----

func.func @extract_scalar_from_vec_2d_f32_outer_dynamic_idx(%arg0: vector<1x16xf32>, %arg1: index) -> f32 {
  %0 = vector.extract %arg0[%arg1, 0]: f32 from vector<1x16xf32>
  return %0 : f32
}

// Lowering supports extracting from multi-dim vectors with dynamic indices
// provided that only the trailing index is dynamic.

// CHECK-LABEL: @extract_scalar_from_vec_2d_f32_outer_dynamic_idx(
//       CHECK:   vector.extract

func.func @extract_scalar_from_vec_2d_f32_outer_dynamic_idx_scalable(%arg0: vector<1x[16]xf32>, %arg1: index) -> f32 {
  %0 = vector.extract %arg0[%arg1, 0]: f32 from vector<1x[16]xf32>
  return %0 : f32
}

// Lowering does not support extracting from multi-dim vectors with non trailing
// dynamic index, but it shouldn't crash.

// CHECK-LABEL: @extract_scalar_from_vec_2d_f32_outer_dynamic_idx_scalable(
//       CHECK:   vector.extract

// -----

func.func @extract_scalar_from_vec_0d_index(%arg0: vector<index>) -> index {
  %0 = vector.extract %arg0[]: index from vector<index>
  return %0 : index
}
// CHECK-LABEL: @extract_scalar_from_vec_0d_index(
//  CHECK-SAME:   %[[A:.*]]: vector<index>)
//       CHECK:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<index> to vector<1xi64>
//       CHECK:   %[[T1:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:   %[[T2:.*]] = llvm.extractelement %[[T0]][%[[T1]] : i64] : vector<1xi64>
//       CHECK:   %[[T3:.*]] = builtin.unrealized_conversion_cast %[[T2]] : i64 to index
//       CHECK:   return %[[T3]] : index

// -----

//===----------------------------------------------------------------------===//
// vector.insertelement
//===----------------------------------------------------------------------===//

func.func @insertelement_into_vec_0d_f32(%arg0: f32, %arg1: vector<f32>) -> vector<f32> {
  %1 = vector.insertelement %arg0, %arg1[] : vector<f32>
  return %1 : vector<f32>
}
// CHECK-LABEL: @insertelement_into_vec_0d_f32
//  CHECK-SAME:   %[[A:.*]]: f32,
//       CHECK:   %[[B:.*]] =  builtin.unrealized_conversion_cast %{{.*}} :
//       CHECK:   vector<f32> to vector<1xf32>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
//       CHECK:   %[[X:.*]] = llvm.insertelement %[[A]], %[[B]][%[[C0]] : {{.*}}] : vector<1xf32>

// -----

func.func @insertelement_into_vec_1d_f32_idx_as_i32(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : i32
  %1 = vector.insertelement %arg0, %arg1[%0 : i32] : vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @insertelement_into_vec_1d_f32_idx_as_i32(
//  CHECK-SAME:   %[[A:.*]]: f32,
//  CHECK-SAME:   %[[B:.*]]: vector<4xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 3 : i32
//       CHECK:   %[[X:.*]] = llvm.insertelement %[[A]], %[[B]][%[[C]] : i32] : vector<4xf32>
//       CHECK:   return %[[X]] : vector<4xf32>

// -----

func.func @insertelement_into_vec_1d_f32_idx_as_i32_scalable(%arg0: f32, %arg1: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = arith.constant 3 : i32
  %1 = vector.insertelement %arg0, %arg1[%0 : i32] : vector<[4]xf32>
  return %1 : vector<[4]xf32>
}
// CHECK-LABEL: @insertelement_into_vec_1d_f32_idx_as_i32_scalable(
//  CHECK-SAME:   %[[A:.*]]: f32,
//  CHECK-SAME:   %[[B:.*]]: vector<[4]xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 3 : i32
//       CHECK:   %[[X:.*]] = llvm.insertelement %[[A]], %[[B]][%[[C]] : i32] : vector<[4]xf32>
//       CHECK:   return %[[X]] : vector<[4]xf32>

// -----

func.func @insertelement_into_vec_1d_f32_scalable_idx_as_index(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.insertelement %arg0, %arg1[%0 : index] : vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @insertelement_into_vec_1d_f32_scalable_idx_as_index(
//  CHECK-SAME:   %[[A:.*]]: f32,
//  CHECK-SAME:   %[[B:.*]]: vector<4xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 3 : index
//       CHECK:   %[[I:.*]] = builtin.unrealized_conversion_cast %[[C]] : index to i64
//       CHECK:   %[[X:.*]] = llvm.insertelement %[[A]], %[[B]][%[[I]] : i64] : vector<4xf32>
//       CHECK:   return %[[X]] : vector<4xf32>

// -----

func.func @insertelement_into_vec_1d_f32_scalable_idx_as_index_scalable(%arg0: f32, %arg1: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.insertelement %arg0, %arg1[%0 : index] : vector<[4]xf32>
  return %1 : vector<[4]xf32>
}
// CHECK-LABEL: @insertelement_into_vec_1d_f32_scalable_idx_as_index_scalable(
//  CHECK-SAME:   %[[A:.*]]: f32,
//  CHECK-SAME:   %[[B:.*]]: vector<[4]xf32>)
//       CHECK:   %[[C:.*]] = arith.constant 3 : index
//       CHECK:   %[[I:.*]] = builtin.unrealized_conversion_cast %[[C]] : index to i64
//       CHECK:   %[[X:.*]] = llvm.insertelement %[[A]], %[[B]][%[[I]] : i64] : vector<[4]xf32>
//       CHECK:   return %[[X]] : vector<[4]xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.insert
//===----------------------------------------------------------------------===//

func.func @insert_scalar_into_vec_1d_f32(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.insert %arg0, %arg1[3] : f32 into vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: @insert_scalar_into_vec_1d_f32
//       CHECK:   llvm.mlir.constant(3 : i64) : i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : i64] : vector<4xf32>
//       CHECK:   return {{.*}} : vector<4xf32>

// -----

func.func @insert_scalar_into_vec_1d_f32_scalable(%arg0: f32, %arg1: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = vector.insert %arg0, %arg1[3] : f32 into vector<[4]xf32>
  return %0 : vector<[4]xf32>
}
// CHECK-LABEL: @insert_scalar_into_vec_1d_f32_scalable
//       CHECK:   llvm.mlir.constant(3 : i64) : i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : i64] : vector<[4]xf32>
//       CHECK:   return {{.*}} : vector<[4]xf32>

// -----

func.func @insert_scalar_into_vec_1d_index(%arg0: index, %arg1: vector<4xindex>) -> vector<4xindex> {
  %0 = vector.insert %arg0, %arg1[3] : index into vector<4xindex>
  return %0 : vector<4xindex>
}
// CHECK-LABEL: @insert_scalar_into_vec_1d_index(
//  CHECK-SAME:   %[[A:.*]]: index,
//  CHECK-SAME:   %[[B:.*]]: vector<4xindex>)
//   CHECK-DAG:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
//   CHECK-DAG:   %[[T1:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<4xindex> to vector<4xi64>
//       CHECK:   %[[T3:.*]] = llvm.mlir.constant(3 : i64) : i64
//       CHECK:   %[[T4:.*]] = llvm.insertelement %[[T0]], %[[T1]][%[[T3]] : i64] : vector<4xi64>
//       CHECK:   %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : vector<4xi64> to vector<4xindex>
//       CHECK:   return %[[T5]] : vector<4xindex>

// -----

func.func @insert_scalar_into_vec_1d_index_scalable(%arg0: index, %arg1: vector<[4]xindex>) -> vector<[4]xindex> {
  %0 = vector.insert %arg0, %arg1[3] : index into vector<[4]xindex>
  return %0 : vector<[4]xindex>
}
// CHECK-LABEL: @insert_scalar_into_vec_1d_index_scalable(
//  CHECK-SAME:   %[[A:.*]]: index,
//  CHECK-SAME:   %[[B:.*]]: vector<[4]xindex>)
//   CHECK-DAG:   %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
//   CHECK-DAG:   %[[T1:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<[4]xindex> to vector<[4]xi64>
//       CHECK:   %[[T3:.*]] = llvm.mlir.constant(3 : i64) : i64
//       CHECK:   %[[T4:.*]] = llvm.insertelement %[[T0]], %[[T1]][%[[T3]] : i64] : vector<[4]xi64>
//       CHECK:   %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : vector<[4]xi64> to vector<[4]xindex>
//       CHECK:   return %[[T5]] : vector<[4]xindex>

// -----

func.func @insert_vec_2d_into_vec_3d_f32(%arg0: vector<8x16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3] : vector<8x16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: @insert_vec_2d_into_vec_3d_f32
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x16xf32>

// -----

func.func @insert_vec_2d_into_vec_3d_f32_scalable(%arg0: vector<8x[16]xf32>, %arg1: vector<4x8x[16]xf32>) -> vector<4x8x[16]xf32> {
  %0 = vector.insert %arg0, %arg1[3] : vector<8x[16]xf32> into vector<4x8x[16]xf32>
  return %0 : vector<4x8x[16]xf32>
}
// CHECK-LABEL: @insert_vec_2d_into_vec_3d_f32_scalable
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm.array<4 x array<8 x vector<[16]xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x[16]xf32>

// -----

func.func @insert_vec_1d_into_vec_3d_f32(%arg0: vector<16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7] : vector<16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: @insert_vec_1d_into_vec_3d_f32
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x16xf32>

// -----

func.func @insert_vec_1d_into_vec_3d_f32_scalable(%arg0: vector<[16]xf32>, %arg1: vector<4x8x[16]xf32>) -> vector<4x8x[16]xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7] : vector<[16]xf32> into vector<4x8x[16]xf32>
  return %0 : vector<4x8x[16]xf32>
}
// CHECK-LABEL: @insert_vec_1d_into_vec_3d_f32_scalable
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<[16]xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x[16]xf32>

// -----

func.func @insert_scalar_into_vec_3d_f32(%arg0: f32, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7, 15] : f32 into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: @insert_scalar_into_vec_3d_f32
//       CHECK:   llvm.extractvalue {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   llvm.mlir.constant(15 : i64) : i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : i64] : vector<16xf32>
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<16xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x16xf32>

// -----

func.func @insert_scalar_into_vec_3d_f32_scalable(%arg0: f32, %arg1: vector<4x8x[16]xf32>) -> vector<4x8x[16]xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7, 15] : f32 into vector<4x8x[16]xf32>
  return %0 : vector<4x8x[16]xf32>
}
// CHECK-LABEL: @insert_scalar_into_vec_3d_f32_scalable
//       CHECK:   llvm.extractvalue {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<[16]xf32>>>
//       CHECK:   llvm.mlir.constant(15 : i64) : i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : i64] : vector<[16]xf32>
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm.array<4 x array<8 x vector<[16]xf32>>>
//       CHECK:   return {{.*}} : vector<4x8x[16]xf32>

// -----

func.func @insert_scalar_into_vec_1d_f32_dynamic_idx(%arg0: vector<16xf32>, %arg1: f32, %arg2: index)
                                      -> vector<16xf32> {
  %0 = vector.insert %arg1, %arg0[%arg2]: f32 into vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @insert_scalar_into_vec_1d_f32_dynamic_idx
//  CHECK-SAME:   %[[DST:.+]]: vector<16xf32>, %[[SRC:.+]]: f32, %[[INDEX:.+]]: index
//       CHECK:   %[[UC:.+]] = builtin.unrealized_conversion_cast %[[INDEX]] : index to i64
//       CHECK:   llvm.insertelement %[[SRC]], %[[DST]][%[[UC]] : i64] : vector<16xf32>

// -----

func.func @insert_scalar_into_vec_1d_f32_dynamic_idx_scalable(%arg0: vector<[16]xf32>, %arg1: f32, %arg2: index)
                                      -> vector<[16]xf32> {
  %0 = vector.insert %arg1, %arg0[%arg2]: f32 into vector<[16]xf32>
  return %0 : vector<[16]xf32>
}

// CHECK-LABEL: @insert_scalar_into_vec_1d_f32_dynamic_idx_scalable
//  CHECK-SAME:   %[[DST:.+]]: vector<[16]xf32>, %[[SRC:.+]]: f32, %[[INDEX:.+]]: index
//       CHECK:   %[[UC:.+]] = builtin.unrealized_conversion_cast %[[INDEX]] : index to i64
//       CHECK:   llvm.insertelement %[[SRC]], %[[DST]][%[[UC]] : i64] : vector<[16]xf32>

// -----

func.func @insert_scalar_into_vec_2d_f32_dynamic_idx(%arg0: vector<1x16xf32>, %arg1: f32, %idx: index)
                                        -> vector<1x16xf32> {
  %0 = vector.insert %arg1, %arg0[0, %idx]: f32 into vector<1x16xf32>
  return %0 : vector<1x16xf32>
}

// Multi-dim vectors are not supported but this test shouldn't crash.

// CHECK-LABEL: @insert_scalar_into_vec_2d_f32_dynamic_idx(
//       CHECK:   vector.insert

// -----

func.func @insert_scalar_into_vec_2d_f32_dynamic_idx_scalable(%arg0: vector<1x[16]xf32>, %arg1: f32, %idx: index)
                                        -> vector<1x[16]xf32> {
  %0 = vector.insert %arg1, %arg0[0, %idx]: f32 into vector<1x[16]xf32>
  return %0 : vector<1x[16]xf32>
}

// Multi-dim vectors are not supported but this test shouldn't crash.

// CHECK-LABEL: @insert_scalar_into_vec_2d_f32_dynamic_idx_scalable(
//       CHECK:   vector.insert

// -----

//===----------------------------------------------------------------------===//
// vector.type_cast
//
// TODO: Add tests for for vector.type_cast that would cover scalable vectors
//===----------------------------------------------------------------------===//

func.func @type_cast_f32(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
  return %0 : memref<vector<8x8x8xf32>>
}
// CHECK-LABEL: @type_cast_f32
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue %[[allocated]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64)>
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue %[[aligned]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64)>

// NOTE: No test for scalable vectors - the input memref is fixed size.

// -----

func.func @type_cast_index(%arg0: memref<8x8x8xindex>) -> memref<vector<8x8x8xindex>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xindex> to memref<vector<8x8x8xindex>>
  return %0 : memref<vector<8x8x8xindex>>
}
// CHECK-LABEL: @type_cast_index(
// CHECK-SAME: %[[A:.*]]: memref<8x8x8xindex>)
//       CHECK:   %{{.*}} = builtin.unrealized_conversion_cast %[[A]] : memref<8x8x8xindex> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>

//       CHECK:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64)> to memref<vector<8x8x8xindex>>

// NOTE: No test for scalable vectors - the input memref is fixed size.

// -----

func.func @type_cast_non_zero_addrspace(%arg0: memref<8x8x8xf32, 3>) -> memref<vector<8x8x8xf32>, 3> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32, 3> to memref<vector<8x8x8xf32>, 3>
  return %0 : memref<vector<8x8x8xf32>, 3>
}
// CHECK-LABEL: @type_cast_non_zero_addrspace
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64)>
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue %[[allocated]], {{.*}}[0] : !llvm.struct<(ptr<3>, ptr<3>, i64)>
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue %[[aligned]], {{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64)>
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<3>, ptr<3>, i64)>

// NOTE: No test for scalable vectors - the input memref is fixed size.

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

func.func @print_scalar_i64(%arg0: i64) {
  vector.print %arg0 : i64
  return
}
// CHECK-LABEL: @print_scalar_i64(
// CHECK-SAME: %[[A:.*]]: i64)
//       CHECK:    llvm.call @printI64(%[[A]]) : (i64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_ui64(%arg0: ui64) {
  vector.print %arg0 : ui64
  return
}
// CHECK-LABEL: @print_scalar_ui64(
// CHECK-SAME: %[[A:.*]]: ui64)
//       CHECK:    %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : ui64 to i64
//       CHECK:    llvm.call @printU64(%[[C]]) : (i64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_index(%arg0: index) {
  vector.print %arg0 : index
  return
}
// CHECK-LABEL: @print_scalar_index(
// CHECK-SAME: %[[A:.*]]: index)
//       CHECK:    %[[C:.*]] = builtin.unrealized_conversion_cast %[[A]] : index to i64
//       CHECK:    llvm.call @printU64(%[[C]]) : (i64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_f32(%arg0: f32) {
  vector.print %arg0 : f32
  return
}
// CHECK-LABEL: @print_scalar_f32(
// CHECK-SAME: %[[A:.*]]: f32)
//       CHECK:    llvm.call @printF32(%[[A]]) : (f32) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

func.func @print_scalar_f64(%arg0: f64) {
  vector.print %arg0 : f64
  return
}
// CHECK-LABEL: @print_scalar_f64(
// CHECK-SAME: %[[A:.*]]: f64)
//       CHECK:    llvm.call @printF64(%[[A]]) : (f64) -> ()
//       CHECK:    llvm.call @printNewline() : () -> ()

// -----

// CHECK-LABEL: module {
// CHECK: llvm.func @printString(!llvm.ptr)
// CHECK: llvm.mlir.global private constant @[[GLOBAL_STR:.*]]({{.*}})
// CHECK: @print_string
//       CHECK-NEXT: %[[GLOBAL_ADDR:.*]] = llvm.mlir.addressof @[[GLOBAL_STR]] : !llvm.ptr
//       CHECK-NEXT: %[[STR_PTR:.*]] = llvm.getelementptr %[[GLOBAL_ADDR]][0] : (!llvm.ptr) -> !llvm.ptr
//       CHECK-NEXT: llvm.call @printString(%[[STR_PTR]]) : (!llvm.ptr) -> ()
func.func @print_string() {
  vector.print str "Hello, World!"
  return
}

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
//       CHECK:    %[[T0:.*]] = llvm.mlir.undef : !llvm.array<2 x vector<8xf32>>
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
//       CHECK:    %[[T0:.*]] = llvm.mlir.undef : !llvm.array<2 x vector<[8]xf32>>
//       CHECK:    %[[T1:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<4 x vector<[8]xf32>>
//       CHECK:    %[[T2:.*]] = llvm.insertvalue %[[T1]], %[[T0]][0] : !llvm.array<2 x vector<[8]xf32>>
//       CHECK:    %[[T3:.*]] = llvm.extractvalue %[[A]][3] : !llvm.array<4 x vector<[8]xf32>>
//       CHECK:    %[[T4:.*]] = llvm.insertvalue %[[T3]], %[[T2]][1] : !llvm.array<2 x vector<[8]xf32>>
//       CHECK:    %[[T5:.*]] = builtin.unrealized_conversion_cast %[[T4]] : !llvm.array<2 x vector<[8]xf32>> to vector<2x[8]xf32>
//       CHECK:    return %[[T5]]

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
// vector.reduction
//===----------------------------------------------------------------------===//

func.func @reduce_0d_f32(%arg0: vector<f32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<f32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_0d_f32(
// CHECK-SAME: %[[A:.*]]: vector<f32>)
//      CHECK: %[[CA:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<f32> to vector<1xf32>
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[CA]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<1xf32>) -> f32
//      CHECK: return %[[V]] : f32

// -----

func.func @reduce_f16(%arg0: vector<16xf16>) -> f16 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf16> into f16
  return %0 : f16
}
// CHECK-LABEL: @reduce_f16(
// CHECK-SAME: %[[A:.*]]: vector<16xf16>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f16, vector<16xf16>) -> f16
//      CHECK: return %[[V]] : f16

// -----

func.func @reduce_f16_scalable(%arg0: vector<[16]xf16>) -> f16 {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xf16> into f16
  return %0 : f16
}
// CHECK-LABEL: @reduce_f16_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf16>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f16, vector<[16]xf16>) -> f16
//      CHECK: return %[[V]] : f16

// -----

func.func @reduce_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<16xf32>) -> f32
//      CHECK: return %[[V]] : f32

// -----

func.func @reduce_f32_scalable(%arg0: vector<[16]xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_f32_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<[16]xf32>) -> f32
//      CHECK: return %[[V]] : f32

// -----

func.func @reduce_f64(%arg0: vector<16xf64>) -> f64 {
  %0 = vector.reduction <add>, %arg0 : vector<16xf64> into f64
  return %0 : f64
}
// CHECK-LABEL: @reduce_f64(
// CHECK-SAME: %[[A:.*]]: vector<16xf64>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<16xf64>) -> f64
//      CHECK: return %[[V]] : f64

// -----

func.func @reduce_f64_scalable(%arg0: vector<[16]xf64>) -> f64 {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xf64> into f64
  return %0 : f64
}
// CHECK-LABEL: @reduce_f64_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf64>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<[16]xf64>) -> f64
//      CHECK: return %[[V]] : f64

// -----

func.func @reduce_i8(%arg0: vector<16xi8>) -> i8 {
  %0 = vector.reduction <add>, %arg0 : vector<16xi8> into i8
  return %0 : i8
}
// CHECK-LABEL: @reduce_i8(
// CHECK-SAME: %[[A:.*]]: vector<16xi8>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i8

// -----

func.func @reduce_i8_scalable(%arg0: vector<[16]xi8>) -> i8 {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xi8> into i8
  return %0 : i8
}
// CHECK-LABEL: @reduce_i8_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xi8>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i8

// -----

func.func @reduce_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <add>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_i32(
// CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i32

// -----

func.func @reduce_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_i32_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i32

// -----

func.func @reduce_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <add>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.add %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <add>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.add %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_mul_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <mul>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_mul_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.mul"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_mul_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <mul>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_mul_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.mul"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_mul_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <mul>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_mul_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.mul"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.mul %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_mul_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <mul>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_mul_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.mul"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.mul %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_fmaximum_f32(%arg0: vector<16xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <maximumf>, %arg0, %arg1 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fmaximum_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fmaximum(%[[A]]) : (vector<16xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.maximum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fmaximum_f32_scalable(%arg0: vector<[16]xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <maximumf>, %arg0, %arg1 : vector<[16]xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fmaximum_f32_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fmaximum(%[[A]]) : (vector<[16]xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.maximum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fminimum_f32(%arg0: vector<16xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <minimumf>, %arg0, %arg1 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fminimum_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fminimum(%[[A]]) : (vector<16xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.minimum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fminimum_f32_scalable(%arg0: vector<[16]xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <minimumf>, %arg0, %arg1 : vector<[16]xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fminimum_f32_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fminimum(%[[A]]) : (vector<[16]xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.minimum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fmax_f32(%arg0: vector<16xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <maxnumf>, %arg0, %arg1 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fmax_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fmax(%[[A]]) : (vector<16xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.maxnum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fmax_f32_scalable(%arg0: vector<[16]xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <maxnumf>, %arg0, %arg1 : vector<[16]xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fmax_f32_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fmax(%[[A]]) : (vector<[16]xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.maxnum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fmin_f32(%arg0: vector<16xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <minnumf>, %arg0, %arg1 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fmin_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fmin(%[[A]]) : (vector<16xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.minnum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_fmin_f32_scalable(%arg0: vector<[16]xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <minnumf>, %arg0, %arg1 : vector<[16]xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: @reduce_fmin_f32_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xf32>, %[[B:.*]]: f32)
//      CHECK: %[[V:.*]] = llvm.intr.vector.reduce.fmin(%[[A]]) : (vector<[16]xf32>) -> f32
//      CHECK: %[[R:.*]] = llvm.intr.minnum(%[[V]], %[[B]]) : (f32, f32) -> f32
//      CHECK: return %[[R]] : f32

// -----

func.func @reduce_minui_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <minui>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minui_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.umin"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minui_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <minui>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minui_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.umin"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minui_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <minui>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minui_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.umin"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "ule" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minui_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <minui>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minui_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.umin"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "ule" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxui_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <maxui>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxui_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.umax"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxui_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <maxui>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxui_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.umax"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxui_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <maxui>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxui_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.umax"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "uge" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxui_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <maxui>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxui_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.umax"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "uge" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minsi_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <minsi>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minsi_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.smin"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minsi_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <minsi>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minsi_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.smin"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minsi_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <minsi>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minsi_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.smin"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "sle" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_minsi_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <minsi>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_minsi_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.smin"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "sle" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxsi_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <maxsi>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxsi_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.smax"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxsi_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <maxsi>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxsi_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.smax"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxsi_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <maxsi>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxsi_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.smax"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "sge" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_maxsi_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <maxsi>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_maxsi_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.smax"(%[[A]])
//       CHECK: %[[S:.*]] = llvm.icmp "sge" %[[ACC]], %[[R]]
//       CHECK: %[[V:.*]] = llvm.select %[[S]], %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_and_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <and>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_and_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.and"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_and_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <and>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_and_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.and"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_and_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <and>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_and_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.and"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.and %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_and_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <and>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_and_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.and"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.and %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_or_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <or>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_or_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.or"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_or_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <or>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_or_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.or"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_or_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <or>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_or_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.or"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.or %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_or_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <or>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_or_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.or"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.or %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_xor_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction <xor>, %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_xor_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.xor"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_xor_i32_scalable(%arg0: vector<[16]xi32>) -> i32 {
  %0 = vector.reduction <xor>, %arg0 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_xor_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>)
//       CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.xor"(%[[A]])
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_xor_acc_i32(%arg0: vector<16xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <xor>, %arg0, %arg1 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_xor_acc_i32(
//  CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.xor"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.xor %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_xor_acc_i32_scalable(%arg0: vector<[16]xi32>, %arg1 : i32) -> i32 {
  %0 = vector.reduction <xor>, %arg0, %arg1 : vector<[16]xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: @reduce_xor_acc_i32_scalable(
//  CHECK-SAME: %[[A:.*]]: vector<[16]xi32>, %[[ACC:.*]]: i32)
//       CHECK: %[[R:.*]] = "llvm.intr.vector.reduce.xor"(%[[A]])
//       CHECK: %[[V:.*]] = llvm.xor %[[ACC]], %[[R]]
//       CHECK: return %[[V]] : i32

// -----

func.func @reduce_i64(%arg0: vector<16xi64>) -> i64 {
  %0 = vector.reduction <add>, %arg0 : vector<16xi64> into i64
  return %0 : i64
}
// CHECK-LABEL: @reduce_i64(
// CHECK-SAME: %[[A:.*]]: vector<16xi64>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i64

// -----

func.func @reduce_i64_scalable(%arg0: vector<[16]xi64>) -> i64 {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xi64> into i64
  return %0 : i64
}
// CHECK-LABEL: @reduce_i64_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xi64>)
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.add"(%[[A]])
//      CHECK: return %[[V]] : i64

// -----

func.func @reduce_index(%arg0: vector<16xindex>) -> index {
  %0 = vector.reduction <add>, %arg0 : vector<16xindex> into index
  return %0 : index
}
// CHECK-LABEL: @reduce_index(
// CHECK-SAME: %[[A:.*]]: vector<16xindex>)
//      CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<16xindex> to vector<16xi64>
//      CHECK: %[[T1:.*]] = "llvm.intr.vector.reduce.add"(%[[T0]])
//      CHECK: %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : i64 to index
//      CHECK: return %[[T2]] : index

// -----

func.func @reduce_index_scalable(%arg0: vector<[16]xindex>) -> index {
  %0 = vector.reduction <add>, %arg0 : vector<[16]xindex> into index
  return %0 : index
}
// CHECK-LABEL: @reduce_index_scalable(
// CHECK-SAME: %[[A:.*]]: vector<[16]xindex>)
//      CHECK: %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<[16]xindex> to vector<[16]xi64>
//      CHECK: %[[T1:.*]] = "llvm.intr.vector.reduce.add"(%[[T0]])
//      CHECK: %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : i64 to index
//      CHECK: return %[[T2]] : index

// -----

//===----------------------------------------------------------------------===//
// vector.matrix_multiply
//===----------------------------------------------------------------------===//

//                          4x16                16x3               4x3
func.func @matrix_ops(%A: vector<64xf64>, %B: vector<48xf64>) -> vector<12xf64> {
  %C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
    (vector<64xf64>, vector<48xf64>) -> vector<12xf64>
  return %C: vector<12xf64>
}
// CHECK-LABEL: @matrix_ops
//       CHECK:   llvm.intr.matrix.multiply %{{.*}}, %{{.*}} {
//  CHECK-SAME: lhs_columns = 16 : i32, lhs_rows = 4 : i32, rhs_columns = 3 : i32
//  CHECK-SAME: } : (vector<64xf64>, vector<48xf64>) -> vector<12xf64>

// -----

func.func @matrix_ops_index(%A: vector<64xindex>, %B: vector<48xindex>) -> vector<12xindex> {
  %C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
    (vector<64xindex>, vector<48xindex>) -> vector<12xindex>
  return %C: vector<12xindex>
}
// CHECK-LABEL: @matrix_ops_index
//       CHECK:   llvm.intr.matrix.multiply %{{.*}}, %{{.*}} {
//  CHECK-SAME: lhs_columns = 16 : i32, lhs_rows = 4 : i32, rhs_columns = 3 : i32
//  CHECK-SAME: } : (vector<64xi64>, vector<48xi64>) -> vector<12xi64>

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
// CHECK: %[[VAL_0:.*]] = arith.constant dense<[true, true, false, false]> : vector<4xi1>
// CHECK: %[[VAL_1:.*]] = arith.constant dense<false> : vector<4x4xi1>
// CHECK: %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : vector<4x4xi1> to !llvm.array<4 x vector<4xi1>>
// CHECK: %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][0] : !llvm.array<4 x vector<4xi1>>
// CHECK: %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][1] : !llvm.array<4 x vector<4xi1>>
// CHECK: %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_4]] : !llvm.array<4 x vector<4xi1>> to vector<4x4xi1>
// CHECK: return %[[VAL_5]] : vector<4x4xi1>

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
// vector.transpose
//===----------------------------------------------------------------------===//

func.func @transpose_0d(%arg0: vector<f32>) -> vector<f32> {
  %0 = vector.transpose %arg0, [] : vector<f32> to vector<f32>
  return %0 : vector<f32>
}

// CHECK-LABEL: func @transpose_0d
// CHECK-SAME:  %[[A:.*]]: vector<f32>
// CHECK:       return %[[A]] : vector<f32>

// -----

//===----------------------------------------------------------------------===//
// vector.flat_transpose
//===----------------------------------------------------------------------===//

func.func @flat_transpose(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 }
     : vector<16xf32> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @flat_transpose
// CHECK-SAME:  %[[A:.*]]: vector<16xf32>
// CHECK:       %[[T:.*]] = llvm.intr.matrix.transpose %[[A]]
// CHECK-SAME:      {columns = 4 : i32, rows = 4 : i32} :
// CHECK-SAME:      vector<16xf32> into vector<16xf32>
// CHECK:       return %[[T]] : vector<16xf32>

// -----

func.func @flat_transpose_index(%arg0: vector<16xindex>) -> vector<16xindex> {
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 }
     : vector<16xindex> -> vector<16xindex>
  return %0 : vector<16xindex>
}
// CHECK-LABEL: func @flat_transpose_index
// CHECK-SAME:  %[[A:.*]]: vector<16xindex>
// CHECK:       %[[T0:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<16xindex> to vector<16xi64>
// CHECK:       %[[T1:.*]] = llvm.intr.matrix.transpose %[[T0]]
// CHECK-SAME:      {columns = 4 : i32, rows = 4 : i32} :
// CHECK-SAME:      vector<16xi64> into vector<16xi64>
// CHECK:       %[[T2:.*]] = builtin.unrealized_conversion_cast %[[T1]] : vector<16xi64> to vector<16xindex>
// CHECK:       return %[[T2]] : vector<16xindex>

// -----

func.func @flat_transpose(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 }
     : vector<16xf32> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @flat_transpose
// CHECK-SAME:  %[[A:.*]]: vector<16xf32>
// CHECK:       %[[T:.*]] = llvm.intr.matrix.transpose %[[A]]
// CHECK-SAME:      {columns = 4 : i32, rows = 4 : i32} :
// CHECK-SAME:      vector<16xf32> into vector<16xf32>
// CHECK:       return %[[T]] : vector<16xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.load
//===----------------------------------------------------------------------===//

func.func @load(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @load
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[GEP]] {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>

// -----

func.func @load_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<[8]xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// CHECK-LABEL: func @load_scalable
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[GEP]] {alignment = 4 : i64} : !llvm.ptr -> vector<[8]xf32>

// -----

func.func @load_nontemporal(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @load_nontemporal
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[GEP]] {alignment = 4 : i64, nontemporal} : !llvm.ptr -> vector<8xf32>

// -----

func.func @load_nontemporal_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<[8]xf32> {
  %0 = vector.load %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// CHECK-LABEL: func @load_nontemporal_scalable
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[GEP]] {alignment = 4 : i64, nontemporal} : !llvm.ptr -> vector<[8]xf32>

// -----

func.func @load_index(%memref : memref<200x100xindex>, %i : index, %j : index) -> vector<8xindex> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xindex>, vector<8xindex>
  return %0 : vector<8xindex>
}
// CHECK-LABEL: func @load_index
// CHECK: %[[T0:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<8xi64>
// CHECK: %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<8xi64> to vector<8xindex>
// CHECK: return %[[T1]] : vector<8xindex>

// -----

func.func @load_index_scalable(%memref : memref<200x100xindex>, %i : index, %j : index) -> vector<[8]xindex> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xindex>, vector<[8]xindex>
  return %0 : vector<[8]xindex>
}
// CHECK-LABEL: func @load_index_scalable
// CHECK: %[[T0:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<[8]xi64>
// CHECK: %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<[8]xi64> to vector<[8]xindex>
// CHECK: return %[[T1]] : vector<[8]xindex>

// -----

func.func @load_0d(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<f32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<f32>
  return %0 : vector<f32>
}

// CHECK-LABEL: func @load_0d
// CHECK: %[[LOAD:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK: %[[VEC:.*]] = llvm.mlir.undef : vector<1xf32>
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[INSERTED:.*]] = llvm.insertelement %[[LOAD]], %[[VEC]][%[[C0]] : i32] : vector<1xf32>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[INSERTED]] : vector<1xf32> to vector<f32>
// CHECK: return %[[CAST]] : vector<f32>

// -----

//===----------------------------------------------------------------------===//
// vector.store
//===----------------------------------------------------------------------===//

func.func @store(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func @store
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64} :  vector<4xf32>, !llvm.ptr

// -----

func.func @store_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<[4]xf32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<[4]xf32>
  return
}

// CHECK-LABEL: func @store_scalable
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64} :  vector<[4]xf32>, !llvm.ptr

// -----

func.func @store_nontemporal(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func @store_nontemporal
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64, nontemporal} :  vector<4xf32>, !llvm.ptr

// -----

func.func @store_nontemporal_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<[4]xf32>
  vector.store %val, %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<[4]xf32>
  return
}

// CHECK-LABEL: func @store_nontemporal_scalable
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]  : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}  : i64
// CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64, nontemporal} :  vector<[4]xf32>, !llvm.ptr

// -----

func.func @store_index(%memref : memref<200x100xindex>, %i : index, %j : index) {
  %val = arith.constant dense<11> : vector<4xindex>
  vector.store %val, %memref[%i, %j] : memref<200x100xindex>, vector<4xindex>
  return
}
// CHECK-LABEL: func @store_index
// CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<4xi64>, !llvm.ptr

// -----

func.func @store_index_scalable(%memref : memref<200x100xindex>, %i : index, %j : index) {
  %val = arith.constant dense<11> : vector<[4]xindex>
  vector.store %val, %memref[%i, %j] : memref<200x100xindex>, vector<[4]xindex>
  return
}
// CHECK-LABEL: func @store_index_scalable
// CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<[4]xi64>, !llvm.ptr

// -----

func.func @store_0d(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<f32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<f32>
  return
}

// CHECK-LABEL: func @store_0d
// CHECK: %[[VAL:.*]] = arith.constant dense<1.100000e+01> : vector<f32>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[VAL]] : vector<f32> to vector<1xf32>
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[EXTRACTED:.*]] = llvm.extractelement %[[CAST]][%[[C0]] : i64] : vector<1xf32>
// CHECK: memref.store %[[EXTRACTED]], %{{.*}}[%{{.*}}, %{{.*}}]

// -----

//===----------------------------------------------------------------------===//
// vector.maskedload
//===----------------------------------------------------------------------===//

func.func @masked_load(%arg0: memref<?xf32>, %arg1: vector<16xi1>, %arg2: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @masked_load
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[L:.*]] = llvm.intr.masked.load %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
// CHECK: return %[[L]] : vector<16xf32>

// -----

func.func @masked_load_scalable(%arg0: memref<?xf32>, %arg1: vector<[16]xi1>, %arg2: vector<[16]xf32>) -> vector<[16]xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<[16]xi1>, vector<[16]xf32> into vector<[16]xf32>
  return %0 : vector<[16]xf32>
}

// CHECK-LABEL: func @masked_load_scalable
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[L:.*]] = llvm.intr.masked.load %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.ptr, vector<[16]xi1>, vector<[16]xf32>) -> vector<[16]xf32>
// CHECK: return %[[L]] : vector<[16]xf32>

// -----

func.func @masked_load_index(%arg0: memref<?xindex>, %arg1: vector<16xi1>, %arg2: vector<16xindex>) -> vector<16xindex> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<16xi1>, vector<16xindex> into vector<16xindex>
  return %0 : vector<16xindex>
}
// CHECK-LABEL: func @masked_load_index
// CHECK: %{{.*}} = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xi64>) -> vector<16xi64>

// -----

func.func @masked_load_index_scalable(%arg0: memref<?xindex>, %arg1: vector<[16]xi1>, %arg2: vector<[16]xindex>) -> vector<[16]xindex> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<[16]xi1>, vector<[16]xindex> into vector<[16]xindex>
  return %0 : vector<[16]xindex>
}
// CHECK-LABEL: func @masked_load_index_scalable
// CHECK: %{{.*}} = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.ptr, vector<[16]xi1>, vector<[16]xi64>) -> vector<[16]xi64>

// -----

//===----------------------------------------------------------------------===//
// vector.maskedstore
//===----------------------------------------------------------------------===//

func.func @masked_store(%arg0: memref<?xf32>, %arg1: vector<16xi1>, %arg2: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: func @masked_store
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.intr.masked.store %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr

// -----

func.func @masked_store_scalable(%arg0: memref<?xf32>, %arg1: vector<[16]xi1>, %arg2: vector<[16]xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<[16]xi1>, vector<[16]xf32>
  return
}

// CHECK-LABEL: func @masked_store_scalable
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.intr.masked.store %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<[16]xf32>, vector<[16]xi1> into !llvm.ptr

// -----

func.func @masked_store_index(%arg0: memref<?xindex>, %arg1: vector<16xi1>, %arg2: vector<16xindex>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<16xi1>, vector<16xindex>
  return
}
// CHECK-LABEL: func @masked_store_index
// CHECK: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<16xi64>, vector<16xi1> into !llvm.ptr

// -----

func.func @masked_store_index_scalable(%arg0: memref<?xindex>, %arg1: vector<[16]xi1>, %arg2: vector<[16]xindex>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<[16]xi1>, vector<[16]xindex>
  return
}
// CHECK-LABEL: func @masked_store_index_scalable
// CHECK: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<[16]xi64>, vector<[16]xi1> into !llvm.ptr

// -----

//===----------------------------------------------------------------------===//
// vector.gather
//===----------------------------------------------------------------------===//

func.func @gather(%arg0: memref<?xf32>, %arg1: vector<3xi32>, %arg2: vector<3xi1>, %arg3: vector<3xf32>) -> vector<3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// CHECK-LABEL: func @gather
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi32>) -> !llvm.vec<3 x ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<3 x ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: return %[[G]] : vector<3xf32>

// -----

func.func @gather_scalable(%arg0: memref<?xf32>, %arg1: vector<[3]xi32>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xf32>) -> vector<[3]xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<[3]xi32>, vector<[3]xi1>, vector<[3]xf32> into vector<[3]xf32>
  return %1 : vector<[3]xf32>
}

// CHECK-LABEL: func @gather_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi32>) -> !llvm.vec<? x 3 x ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<? x 3 x ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: return %[[G]] : vector<[3]xf32>

// -----

func.func @gather_global_memory(%arg0: memref<?xf32, 1>, %arg1: vector<3xi32>, %arg2: vector<3xi1>, %arg3: vector<3xf32>) -> vector<3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32, 1>, vector<3xi32>, vector<3xi1>, vector<3xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// CHECK-LABEL: func @gather_global_memory
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, vector<3xi32>) -> !llvm.vec<3 x ptr<1>>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<3 x ptr<1>>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: return %[[G]] : vector<3xf32>

// -----

func.func @gather_global_memory_scalable(%arg0: memref<?xf32, 1>, %arg1: vector<[3]xi32>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xf32>) -> vector<[3]xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32, 1>, vector<[3]xi32>, vector<[3]xi1>, vector<[3]xf32> into vector<[3]xf32>
  return %1 : vector<[3]xf32>
}

// CHECK-LABEL: func @gather_global_memory_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, vector<[3]xi32>) -> !llvm.vec<? x 3 x ptr<1>>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<? x 3 x ptr<1>>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: return %[[G]] : vector<[3]xf32>

// -----


func.func @gather_index(%arg0: memref<?xindex>, %arg1: vector<3xindex>, %arg2: vector<3xi1>, %arg3: vector<3xindex>) -> vector<3xindex> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<3xindex>, vector<3xi1>, vector<3xindex> into vector<3xindex>
  return %1 : vector<3xindex>
}

// CHECK-LABEL: func @gather_index
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi64>) -> !llvm.vec<3 x ptr>, i64
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.vec<3 x ptr>, vector<3xi1>, vector<3xi64>) -> vector<3xi64>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[G]] : vector<3xi64> to vector<3xindex>

// -----

func.func @gather_index_scalable(%arg0: memref<?xindex>, %arg1: vector<[3]xindex>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xindex>) -> vector<[3]xindex> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<[3]xindex>, vector<[3]xi1>, vector<[3]xindex> into vector<[3]xindex>
  return %1 : vector<[3]xindex>
}

// CHECK-LABEL: func @gather_index_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi64>) -> !llvm.vec<? x 3 x ptr>, i64
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.vec<? x 3 x ptr>, vector<[3]xi1>, vector<[3]xi64>) -> vector<[3]xi64>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[G]] : vector<[3]xi64> to vector<[3]xindex>

// -----

func.func @gather_2d_from_1d(%arg0: memref<?xf32>, %arg1: vector<2x3xi32>, %arg2: vector<2x3xi1>, %arg3: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<2x3xi32>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %1 : vector<2x3xf32>
}

// CHECK-LABEL: func @gather_2d_from_1d
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[I0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<3xi32>>
// CHECK: %[[M0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<3xi1>>
// CHECK: %[[S0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<3xf32>>
// CHECK: %[[P0:.*]] = llvm.getelementptr %[[B]][%[[I0]]] : (!llvm.ptr, vector<3xi32>) -> !llvm.vec<3 x ptr>, f32
// CHECK: %[[G0:.*]] = llvm.intr.masked.gather %[[P0]], %[[M0]], %[[S0]] {alignment = 4 : i32} : (!llvm.vec<3 x ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: %{{.*}} = llvm.insertvalue %[[G0]], %{{.*}}[0] : !llvm.array<2 x vector<3xf32>>
// CHECK: %[[I1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<3xi32>>
// CHECK: %[[M1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<3xi1>>
// CHECK: %[[S1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<3xf32>>
// CHECK: %[[P1:.*]] = llvm.getelementptr %[[B]][%[[I1]]] : (!llvm.ptr, vector<3xi32>) -> !llvm.vec<3 x ptr>, f32
// CHECK: %[[G1:.*]] = llvm.intr.masked.gather %[[P1]], %[[M1]], %[[S1]] {alignment = 4 : i32} : (!llvm.vec<3 x ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: %{{.*}} = llvm.insertvalue %[[G1]], %{{.*}}[1] : !llvm.array<2 x vector<3xf32>>

// -----

func.func @gather_2d_from_1d_scalable(%arg0: memref<?xf32>, %arg1: vector<2x[3]xi32>, %arg2: vector<2x[3]xi1>, %arg3: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<2x[3]xi32>, vector<2x[3]xi1>, vector<2x[3]xf32> into vector<2x[3]xf32>
  return %1 : vector<2x[3]xf32>
}

// CHECK-LABEL: func @gather_2d_from_1d_scalable
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[I0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<[3]xi32>>
// CHECK: %[[M0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<[3]xi1>>
// CHECK: %[[S0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<[3]xf32>>
// CHECK: %[[P0:.*]] = llvm.getelementptr %[[B]][%[[I0]]] : (!llvm.ptr, vector<[3]xi32>) -> !llvm.vec<? x 3 x ptr>, f32
// CHECK: %[[G0:.*]] = llvm.intr.masked.gather %[[P0]], %[[M0]], %[[S0]] {alignment = 4 : i32} : (!llvm.vec<? x 3 x ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: %{{.*}} = llvm.insertvalue %[[G0]], %{{.*}}[0] : !llvm.array<2 x vector<[3]xf32>>
// CHECK: %[[I1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<[3]xi32>>
// CHECK: %[[M1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<[3]xi1>>
// CHECK: %[[S1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<[3]xf32>>
// CHECK: %[[P1:.*]] = llvm.getelementptr %[[B]][%[[I1]]] : (!llvm.ptr, vector<[3]xi32>) -> !llvm.vec<? x 3 x ptr>, f32
// CHECK: %[[G1:.*]] = llvm.intr.masked.gather %[[P1]], %[[M1]], %[[S1]] {alignment = 4 : i32} : (!llvm.vec<? x 3 x ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: %{{.*}} = llvm.insertvalue %[[G1]], %{{.*}}[1] : !llvm.array<2 x vector<[3]xf32>>

// -----

func.func @gather_with_mask(%arg0: memref<?xf32>, %arg1: vector<2x3xi32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.constant_mask [1, 2] : vector<2x3xi1>
  %2 = vector.gather %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x3xi32>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %2 : vector<2x3xf32>
}

// CHECK-LABEL: func @gather_with_mask
// CHECK: %[[G0:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<3 x ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: %[[G1:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<3 x ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>

// -----

func.func @gather_with_mask_scalable(%arg0: memref<?xf32>, %arg1: vector<2x[3]xi32>, %arg2: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = arith.constant 0: index
  // vector.constant_mask only supports 'none set' or 'all set' scalable
  // dimensions, hence [1, 3] rather than [1, 2] as in the example for fixed
  // width vectors above.
  %1 = vector.constant_mask [1, 3] : vector<2x[3]xi1>
  %2 = vector.gather %arg0[%0][%arg1], %1, %arg2 : memref<?xf32>, vector<2x[3]xi32>, vector<2x[3]xi1>, vector<2x[3]xf32> into vector<2x[3]xf32>
  return %2 : vector<2x[3]xf32>
}

// CHECK-LABEL: func @gather_with_mask_scalable
// CHECK: %[[G0:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<? x 3 x ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: %[[G1:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<? x 3 x ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>


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

func.func @gather_1d_from_2d(%arg0: memref<4x4xf32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.gather %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x4xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: func @gather_1d_from_2d
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<4xi32>) -> !llvm.vec<4 x ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<4 x ptr>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
// CHECK: return %[[G]] : vector<4xf32>

// -----

func.func @gather_1d_from_2d_scalable(%arg0: memref<4x?xf32>, %arg1: vector<[4]xi32>, %arg2: vector<[4]xi1>, %arg3: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.gather %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x?xf32>, vector<[4]xi32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
  return %1 : vector<[4]xf32>
}

// CHECK-LABEL: func @gather_1d_from_2d_scalable
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<[4]xi32>) -> !llvm.vec<? x 4 x ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.vec<? x 4 x ptr>, vector<[4]xi1>, vector<[4]xf32>) -> vector<[4]xf32>
// CHECK: return %[[G]] : vector<[4]xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.scatter
//===----------------------------------------------------------------------===//

func.func @scatter(%arg0: memref<?xf32>, %arg1: vector<3xi32>, %arg2: vector<3xi1>, %arg3: vector<3xf32>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
  return
}

// CHECK-LABEL: func @scatter
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi32>) -> !llvm.vec<3 x ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<3xf32>, vector<3xi1> into !llvm.vec<3 x ptr>

// -----

func.func @scatter_scalable(%arg0: memref<?xf32>, %arg1: vector<[3]xi32>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xf32>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<[3]xi32>, vector<[3]xi1>, vector<[3]xf32>
  return
}

// CHECK-LABEL: func @scatter_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi32>) -> !llvm.vec<? x 3 x ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<[3]xf32>, vector<[3]xi1> into !llvm.vec<? x 3 x ptr>

// -----

func.func @scatter_index(%arg0: memref<?xindex>, %arg1: vector<3xindex>, %arg2: vector<3xi1>, %arg3: vector<3xindex>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<3xindex>, vector<3xi1>, vector<3xindex>
  return
}

// CHECK-LABEL: func @scatter_index
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi64>) -> !llvm.vec<3 x ptr>, i64
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 8 : i32} : vector<3xi64>, vector<3xi1> into !llvm.vec<3 x ptr>

// -----

func.func @scatter_index_scalable(%arg0: memref<?xindex>, %arg1: vector<[3]xindex>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xindex>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<[3]xindex>, vector<[3]xi1>, vector<[3]xindex>
  return
}

// CHECK-LABEL: func @scatter_index_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi64>) -> !llvm.vec<? x 3 x ptr>, i64
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 8 : i32} : vector<[3]xi64>, vector<[3]xi1> into !llvm.vec<? x 3 x ptr>

// -----

func.func @scatter_1d_into_2d(%arg0: memref<4x4xf32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: vector<4xf32>) {
  %0 = arith.constant 3 : index
  vector.scatter %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x4xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL: func @scatter_1d_into_2d
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<4xi32>) -> !llvm.vec<4 x ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<4xf32>, vector<4xi1> into !llvm.vec<4 x ptr>

// -----

func.func @scatter_1d_into_2d_scalable(%arg0: memref<4x?xf32>, %arg1: vector<[4]xi32>, %arg2: vector<[4]xi1>, %arg3: vector<[4]xf32>) {
  %0 = arith.constant 3 : index
  vector.scatter %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x?xf32>, vector<[4]xi32>, vector<[4]xi1>, vector<[4]xf32>
  return
}

// CHECK-LABEL: func @scatter_1d_into_2d_scalable
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<[4]xi32>) -> !llvm.vec<? x 4 x ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<[4]xf32>, vector<[4]xi1> into !llvm.vec<? x 4 x ptr>

// -----

//===----------------------------------------------------------------------===//
// vector.expandload
//===----------------------------------------------------------------------===//

func.func @expand_load_op(%arg0: memref<?xf32>, %arg1: vector<11xi1>, %arg2: vector<11xf32>) -> vector<11xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.expandload %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<11xi1>, vector<11xf32> into vector<11xf32>
  return %0 : vector<11xf32>
}

// CHECK-LABEL: func @expand_load_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[E:.*]] = "llvm.intr.masked.expandload"(%[[P]], %{{.*}}, %{{.*}}) : (!llvm.ptr, vector<11xi1>, vector<11xf32>) -> vector<11xf32>
// CHECK: return %[[E]] : vector<11xf32>

// -----

func.func @expand_load_op_index(%arg0: memref<?xindex>, %arg1: vector<11xi1>, %arg2: vector<11xindex>) -> vector<11xindex> {
  %c0 = arith.constant 0: index
  %0 = vector.expandload %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<11xi1>, vector<11xindex> into vector<11xindex>
  return %0 : vector<11xindex>
}
// CHECK-LABEL: func @expand_load_op_index
// CHECK: %{{.*}} = "llvm.intr.masked.expandload"(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, vector<11xi1>, vector<11xi64>) -> vector<11xi64>

// -----

//===----------------------------------------------------------------------===//
// vector.compressstore
//===----------------------------------------------------------------------===//

func.func @compress_store_op(%arg0: memref<?xf32>, %arg1: vector<11xi1>, %arg2: vector<11xf32>) {
  %c0 = arith.constant 0: index
  vector.compressstore %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<11xi1>, vector<11xf32>
  return
}

// CHECK-LABEL: func @compress_store_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: "llvm.intr.masked.compressstore"(%{{.*}}, %[[P]], %{{.*}}) : (vector<11xf32>, !llvm.ptr, vector<11xi1>) -> ()

// -----

func.func @compress_store_op_index(%arg0: memref<?xindex>, %arg1: vector<11xi1>, %arg2: vector<11xindex>) {
  %c0 = arith.constant 0: index
  vector.compressstore %arg0[%c0], %arg1, %arg2 : memref<?xindex>, vector<11xi1>, vector<11xindex>
  return
}
// CHECK-LABEL: func @compress_store_op_index
// CHECK: "llvm.intr.masked.compressstore"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<11xi64>, !llvm.ptr, vector<11xi1>) -> ()

// -----

//===----------------------------------------------------------------------===//
// vector.splat
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @splat_0d
// CHECK-SAME: %[[ELT:.*]]: f32
func.func @splat_0d(%elt: f32) -> vector<f32> {
  %v = vector.splat %elt : vector<f32>
  return %v : vector<f32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.undef : vector<1xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<1xf32>
// CHECK-NEXT: %[[VCAST:[0-9]+]] = builtin.unrealized_conversion_cast %[[V]] : vector<1xf32> to vector<f32>
// CHECK-NEXT: return %[[VCAST]] : vector<f32>

// -----

// CHECK-LABEL: @splat
// CHECK-SAME: %[[VEC:[0-9a-zA-Z]+]]: vector<4xf32>
// CHECK-SAME: %[[ELT:[0-9a-zA-Z]+]]: f32
func.func @splat(%vec: vector<4xf32>, %elt: f32) -> vector<4xf32> {
  %vb = vector.splat %elt : vector<4xf32>
  %r = arith.mulf %vec, %vb : vector<4xf32>
  return %r : vector<4xf32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.undef : vector<4xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<4xf32>
// CHECK-NEXT: %[[SPLAT:[0-9]+]] = llvm.shufflevector %[[V]], %[[UNDEF]] [0, 0, 0, 0]
// CHECK-NEXT: %[[SCALE:[0-9]+]] = arith.mulf %[[VEC]], %[[SPLAT]] : vector<4xf32>
// CHECK-NEXT: return %[[SCALE]] : vector<4xf32>

// -----

// CHECK-LABEL: @splat_scalable
// CHECK-SAME: %[[VEC:[0-9a-zA-Z]+]]: vector<[4]xf32>
// CHECK-SAME: %[[ELT:[0-9a-zA-Z]+]]: f32
func.func @splat_scalable(%vec: vector<[4]xf32>, %elt: f32) -> vector<[4]xf32> {
  %vb = vector.splat %elt : vector<[4]xf32>
  %r = arith.mulf %vec, %vb : vector<[4]xf32>
  return %r : vector<[4]xf32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.undef : vector<[4]xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<[4]xf32>
// CHECK-NEXT: %[[SPLAT:[0-9]+]] = llvm.shufflevector %[[V]], %[[UNDEF]] [0, 0, 0, 0]
// CHECK-NEXT: %[[SCALE:[0-9]+]] = arith.mulf %[[VEC]], %[[SPLAT]] : vector<[4]xf32>
// CHECK-NEXT: return %[[SCALE]] : vector<[4]xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.scalable_insert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @scalable_insert
// CHECK-SAME: %[[SUB:.*]]: vector<4xf32>, %[[SV:.*]]: vector<[4]xf32>
func.func @scalable_insert(%sub: vector<4xf32>, %dsv: vector<[4]xf32>) -> vector<[4]xf32> {
  // CHECK-NEXT: %[[TMP:.*]] = llvm.intr.vector.insert %[[SUB]], %[[SV]][0] : vector<4xf32> into vector<[4]xf32>
  %0 = vector.scalable.insert %sub, %dsv[0] : vector<4xf32> into vector<[4]xf32>
  // CHECK-NEXT: llvm.intr.vector.insert %[[SUB]], %[[TMP]][4] : vector<4xf32> into vector<[4]xf32>
  %1 = vector.scalable.insert %sub, %0[4] : vector<4xf32> into vector<[4]xf32>
  return %1 : vector<[4]xf32>
}

// -----

//===----------------------------------------------------------------------===//
// vector.scalable_extract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @scalable_extract
// CHECK-SAME: %[[VEC:.*]]: vector<[4]xf32>
func.func @scalable_extract(%vec: vector<[4]xf32>) -> vector<8xf32> {
  // CHECK-NEXT: %{{.*}} = llvm.intr.vector.extract %[[VEC]][0] : vector<8xf32> from vector<[4]xf32>
  %0 = vector.scalable.extract %vec[0] : vector<8xf32> from vector<[4]xf32>
  return %0 : vector<8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// vector.interleave
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @interleave_0d
//  CHECK-SAME:     %[[LHS:.*]]: vector<i8>, %[[RHS:.*]]: vector<i8>)
func.func @interleave_0d(%a: vector<i8>, %b: vector<i8>) -> vector<2xi8> {
  // CHECK-DAG: %[[LHS_RANK1:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : vector<i8> to vector<1xi8>
  // CHECK-DAG: %[[RHS_RANK1:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : vector<i8> to vector<1xi8>
  // CHECK: %[[ZIP:.*]] = llvm.shufflevector %[[LHS_RANK1]], %[[RHS_RANK1]] [0, 1] : vector<1xi8>
  // CHECK: return %[[ZIP]]
  %0 = vector.interleave %a, %b : vector<i8> -> vector<2xi8>
  return %0 : vector<2xi8>
}

// -----

// CHECK-LABEL: @interleave_1d
//  CHECK-SAME:     %[[LHS:.*]]: vector<8xf32>, %[[RHS:.*]]: vector<8xf32>)
func.func @interleave_1d(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<16xf32> {
  // CHECK: %[[ZIP:.*]] = llvm.shufflevector %[[LHS]], %[[RHS]] [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>
  // CHECK: return %[[ZIP]]
  %0 = vector.interleave %a, %b : vector<8xf32> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// CHECK-LABEL: @interleave_1d_scalable
//  CHECK-SAME:     %[[LHS:.*]]: vector<[4]xi32>, %[[RHS:.*]]: vector<[4]xi32>)
func.func @interleave_1d_scalable(%a: vector<[4]xi32>, %b: vector<[4]xi32>) -> vector<[8]xi32> {
  // CHECK: %[[ZIP:.*]] = "llvm.intr.vector.interleave2"(%[[LHS]], %[[RHS]]) : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[8]xi32>
  // CHECK: return %[[ZIP]]
  %0 = vector.interleave %a, %b : vector<[4]xi32> -> vector<[8]xi32>
  return %0 : vector<[8]xi32>
}

// -----

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

// CHECK-LABEL: @deinterleave_1d
// CHECK-SAME:  (%[[ARG:.*]]: vector<4xi32>) -> (vector<2xi32>, vector<2xi32>)
func.func @deinterleave_1d(%arg: vector<4xi32>) -> (vector<2xi32>, vector<2xi32>) {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: llvm.shufflevector %[[ARG]], %[[POISON]] [0, 2] : vector<4xi32>
  // CHECK: llvm.shufflevector %[[ARG]], %[[POISON]] [1, 3] : vector<4xi32>
  %0, %1 = vector.deinterleave %arg : vector<4xi32> -> vector<2xi32>
  return %0, %1 : vector<2xi32>, vector<2xi32>
}

// -----

// CHECK-LABEL: @deinterleave_1d_scalable
// CHECK-SAME:  %[[ARG:.*]]: vector<[4]xi32>) -> (vector<[2]xi32>, vector<[2]xi32>)
func.func @deinterleave_1d_scalable(%arg: vector<[4]xi32>) -> (vector<[2]xi32>, vector<[2]xi32>) {
    // CHECK: %[[RES:.*]] = "llvm.intr.vector.deinterleave2"(%[[ARG]]) : (vector<[4]xi32>) -> !llvm.struct<(vector<[2]xi32>, vector<[2]xi32>)>
    // CHECK: llvm.extractvalue %[[RES]][0] : !llvm.struct<(vector<[2]xi32>, vector<[2]xi32>)>
    // CHECK: llvm.extractvalue %[[RES]][1] : !llvm.struct<(vector<[2]xi32>, vector<[2]xi32>)>
    %0, %1 = vector.deinterleave %arg : vector<[4]xi32> -> vector<[2]xi32>
    return %0, %1 : vector<[2]xi32>, vector<[2]xi32>
}

// -----

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
// vector.from_elements
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @from_elements_1d(
//  CHECK-SAME:     %[[ARG_0:.*]]: f32, %[[ARG_1:.*]]: f32)
//       CHECK:   %[[UNDEF:.*]] = llvm.mlir.undef : vector<3xf32>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:   %[[INSERT0:.*]] = llvm.insertelement %[[ARG_0]], %[[UNDEF]][%[[C0]] : i64] : vector<3xf32>
//       CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:   %[[INSERT1:.*]] = llvm.insertelement %[[ARG_1]], %[[INSERT0]][%[[C1]] : i64] : vector<3xf32>
//       CHECK:   %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
//       CHECK:   %[[INSERT2:.*]] = llvm.insertelement %[[ARG_0]], %[[INSERT1]][%[[C2]] : i64] : vector<3xf32>
//       CHECK:   return %[[INSERT2]]
func.func @from_elements_1d(%arg0: f32, %arg1: f32) -> vector<3xf32> {
  %0 = vector.from_elements %arg0, %arg1, %arg0 : vector<3xf32>
  return %0 : vector<3xf32>
}

// -----

// CHECK-LABEL: func.func @from_elements_0d(
//  CHECK-SAME:     %[[ARG_0:.*]]: f32)
//       CHECK:   %[[UNDEF:.*]] = llvm.mlir.undef : vector<1xf32>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:   %[[INSERT0:.*]] = llvm.insertelement %[[ARG_0]], %[[UNDEF]][%[[C0]] : i64] : vector<1xf32>
//       CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[INSERT0]] : vector<1xf32> to vector<f32>
//       CHECK:   return %[[CAST]]
func.func @from_elements_0d(%arg0: f32) -> vector<f32> {
  %0 = vector.from_elements %arg0 : vector<f32>
  return %0 : vector<f32>
}

// -----

//===----------------------------------------------------------------------===//
// vector.step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @step
// CHECK: %[[CST:.+]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK: return %[[CST]] : vector<4xindex>
func.func @step() -> vector<4xindex> {
  %0 = vector.step : vector<4xindex>
  return %0 : vector<4xindex>
}

// -----

// CHECK-LABEL: @step_scalable
// CHECK: %[[STEPVECTOR:.*]] = llvm.intr.stepvector : vector<[4]xi64>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[STEPVECTOR]] : vector<[4]xi64> to vector<[4]xindex>
// CHECK: return %[[CAST]] : vector<[4]xindex>
func.func @step_scalable() -> vector<[4]xindex> {
  %0 = vector.step : vector<[4]xindex>
  return %0 : vector<[4]xindex>
}
