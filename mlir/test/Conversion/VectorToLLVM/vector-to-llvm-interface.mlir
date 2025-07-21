// RUN: mlir-opt --convert-to-llvm="filter-dialects=vector" --split-input-file %s | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s

//===========================================================================//
// Basic tests for Vector-to-LLVM conversion
//
// These examples are meant to be convertible to LLVM with:
//  * `populateVectorToLLVMConversionPatterns`,
// i.e. no other patterns should be required.
//===========================================================================//

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

//===----------------------------------------------------------------------===//
// vector.broadcast
//===----------------------------------------------------------------------===//

func.func @broadcast_vec0d_from_vec0d(%arg0: vector<f32>) -> vector<f32> {
  %0 = vector.broadcast %arg0 : vector<f32> to vector<f32>
  return %0 : vector<f32>
}
// CHECK-LABEL: @broadcast_vec0d_from_vec0d(
// CHECK-SAME:  %[[A:.*]]: vector<f32>)
// CHECK:       return %[[A]] : vector<f32>

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

// CHECK-LABEL: @broadcast_vec0d_from_scalar
// CHECK-SAME: %[[ELT:.*]]: f32
func.func @broadcast_vec0d_from_scalar(%elt: f32) -> vector<f32> {
  %v = vector.broadcast %elt : f32 to vector<f32>
  return %v : vector<f32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.poison : vector<1xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<1xf32>
// CHECK-NEXT: %[[VCAST:[0-9]+]] = builtin.unrealized_conversion_cast %[[V]] : vector<1xf32> to vector<f32>
// CHECK-NEXT: return %[[VCAST]] : vector<f32>

// -----

// CHECK-LABEL: @broadcast_vec1d_from_scalar
// CHECK-SAME: %[[VEC:[0-9a-zA-Z]+]]: vector<4xf32>
// CHECK-SAME: %[[ELT:[0-9a-zA-Z]+]]: f32
func.func @broadcast_vec1d_from_scalar(%vec: vector<4xf32>, %elt: f32) -> vector<4xf32> {
  %vb = vector.broadcast %elt : f32 to vector<4xf32>
  return %vb : vector<4xf32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.poison : vector<4xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<4xf32>
// CHECK-NEXT: %[[SPLAT:[0-9]+]] = llvm.shufflevector %[[V]], %[[UNDEF]] [0, 0, 0, 0]
// CHECK-NEXT: return %[[SPLAT]] : vector<4xf32>

// -----

// CHECK-LABEL: @broadcast_scalable_vec1d_from_scalar
// CHECK-SAME: %[[VEC:[0-9a-zA-Z]+]]: vector<[4]xf32>
// CHECK-SAME: %[[ELT:[0-9a-zA-Z]+]]: f32
func.func @broadcast_scalable_vec1d_from_scalar(%vec: vector<[4]xf32>, %elt: f32) -> vector<[4]xf32> {
  %vb = vector.broadcast %elt : f32 to vector<[4]xf32>
  return %vb : vector<[4]xf32>
}
// CHECK-NEXT: %[[UNDEF:[0-9]+]] = llvm.mlir.poison : vector<[4]xf32>
// CHECK-NEXT: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V:[0-9]+]] = llvm.insertelement %[[ELT]], %[[UNDEF]][%[[ZERO]] : i32] : vector<[4]xf32>
// CHECK-NEXT: %[[SPLAT:[0-9]+]] = llvm.shufflevector %[[V]], %[[UNDEF]] [0, 0, 0, 0]
// CHECK-NEXT: return %[[SPLAT]] : vector<[4]xf32>

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

func.func @shuffle_poison_mask(%arg0: vector<2xf32>, %arg1: vector<2xf32>) -> vector<4xf32> {
  %1 = vector.shuffle %arg0, %arg1 [0, -1, 3, -1] : vector<2xf32>, vector<2xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @shuffle_poison_mask(
//  CHECK-SAME:   %[[A:.*]]: vector<2xf32>, %[[B:.*]]: vector<2xf32>)
//       CHECK:     %[[s:.*]] = llvm.shufflevector %[[A]], %[[B]] [0, -1, 3, -1] : vector<2xf32>

// -----

func.func @shuffle_1D(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<5xf32> {
  %1 = vector.shuffle %arg0, %arg1 [4, 3, 2, 1, 0] : vector<2xf32>, vector<3xf32>
  return %1 : vector<5xf32>
}
// CHECK-LABEL: @shuffle_1D(
// CHECK-SAME: %[[A:.*]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*]]: vector<3xf32>)
//       CHECK:   %[[U0:.*]] = llvm.mlir.poison : vector<5xf32>
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
//       CHECK:   %[[U0:.*]] = llvm.mlir.poison : !llvm.array<3 x vector<4xf32>>
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

func.func @extract_scalar_from_vec_2d_f32_dynamic_idxs_compile_time_const(%arg : vector<32x1xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = vector.extract %arg[%0, %0] : f32 from vector<32x1xf32>
  return %1 : f32
}

// At compile time, since the indices of extractOp are constants,
// they will be collapsed and folded away; therefore, the lowering works.

// CHECK-LABEL: @extract_scalar_from_vec_2d_f32_dynamic_idxs_compile_time_const
//  CHECK-SAME:   %[[ARG:.*]]: vector<32x1xf32>) -> f32 {
//       CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<32x1xf32> to !llvm.array<32 x vector<1xf32>>
//       CHECK:   %[[VEC_0:.*]] = llvm.extractvalue %[[CAST]][0] : !llvm.array<32 x vector<1xf32>>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:   %[[RES:.*]] = llvm.extractelement %[[VEC_0]]{{\[}}%[[C0]] : i64] : vector<1xf32>
//       CHECK:   return %[[RES]] : f32

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

func.func @insert_scalar_into_vec_0d(%src: f32, %dst: vector<f32>) -> vector<f32> {
  %0 = vector.insert %src, %dst[] : f32 into vector<f32>
  return %0 : vector<f32>
}

// CHECK-LABEL: @insert_scalar_into_vec_0d
//       CHECK: llvm.insertelement {{.*}} : vector<1xf32>

// -----

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

// CHECK-LABEL: @insert_scalar_into_vec_2d_f32_dynamic_idx(
//       CHECK:   llvm.extractvalue {{.*}} : !llvm.array<1 x vector<16xf32>>
//       CHECK:   llvm.insertelement {{.*}} : vector<16xf32>
//       CHECK:   llvm.insertvalue {{.*}} : !llvm.array<1 x vector<16xf32>>

// -----

func.func @insert_scalar_into_vec_2d_f32_dynamic_idx_scalable(%arg0: vector<1x[16]xf32>, %arg1: f32, %idx: index)
                                        -> vector<1x[16]xf32> {
  %0 = vector.insert %arg1, %arg0[0, %idx]: f32 into vector<1x[16]xf32>
  return %0 : vector<1x[16]xf32>
}

// CHECK-LABEL: @insert_scalar_into_vec_2d_f32_dynamic_idx_scalable(
//       CHECK:   llvm.extractvalue {{.*}} : !llvm.array<1 x vector<[16]xf32>>
//       CHECK:   llvm.insertelement {{.*}} : vector<[16]xf32>
//       CHECK:   llvm.insertvalue {{.*}} : !llvm.array<1 x vector<[16]xf32>>


// -----

func.func @insert_scalar_into_vec_2d_f32_dynamic_idx_fail(%arg0: vector<2x16xf32>, %arg1: f32, %idx: index)
                                        -> vector<2x16xf32> {
  %0 = vector.insert %arg1, %arg0[%idx, 0]: f32 into vector<2x16xf32>
  return %0 : vector<2x16xf32>
}

// Currently fails to convert because of the dynamic index in non-innermost
// dimension that converts to a llvm.array, as llvm.extractvalue does not
// support dynamic dimensions

// CHECK-LABEL: @insert_scalar_into_vec_2d_f32_dynamic_idx_fail
//       CHECK: vector.insert

// -----

func.func @insert_scalar_from_vec_2d_f32_dynamic_idxs_compile_time_const(%arg : vector<4x1xf32>) -> vector<4x1xf32> {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1.0 : f32
  %res = vector.insert %1, %arg[%0, %0] : f32 into vector<4x1xf32>
  return %res : vector<4x1xf32>
}

// At compile time, since the indices of insertOp are constants,
// they will be collapsed and folded away; therefore, the lowering works.

// CHECK-LABEL: @insert_scalar_from_vec_2d_f32_dynamic_idxs_compile_time_const
//  CHECK-SAME:   %[[ARG:.*]]: vector<4x1xf32>) -> vector<4x1xf32> {
//       CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<4x1xf32> to !llvm.array<4 x vector<1xf32>>
//       CHECK:   %[[C1:.*]] = arith.constant 1.000000e+00 : f32
//       CHECK:   %[[VEC_0:.*]] = llvm.extractvalue %[[CAST]][0] : !llvm.array<4 x vector<1xf32>>
//       CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
//       CHECK:   %[[VEC_1:.*]] = llvm.insertelement %[[C1]], %[[VEC_0]]{{\[}}%[[C0]] : i64] : vector<1xf32>
//       CHECK:   %[[VEC_2:.*]] = llvm.insertvalue %[[VEC_1]], %[[CAST]][0] : !llvm.array<4 x vector<1xf32>>
//       CHECK:   %[[RES:.*]] = builtin.unrealized_conversion_cast %[[VEC_2]] : !llvm.array<4 x vector<1xf32>> to vector<4x1xf32>
//       CHECK:   return %[[RES]] : vector<4x1xf32>

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
//       CHECK:   llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
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
//       CHECK:   llvm.mlir.poison : !llvm.struct<(ptr<3>, ptr<3>, i64)>
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
// CHECK: %[[J:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// CHECK: %[[I:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// CHECK: %[[CAST_MEMREF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<200x100xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[REF:.*]] = llvm.extractvalue %[[CAST_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %[[I]], %[[C100]] : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %[[J]] : i64
// CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[REF]][%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[LOAD:.*]] = llvm.load %[[ADDR]] {alignment = 4 : i64} : !llvm.ptr -> vector<1xf32>
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[LOAD]] : vector<1xf32> to vector<f32>
// CHECK: return %[[RES]] : vector<f32>

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
// CHECK: %[[J:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// CHECK: %[[I:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// CHECK: %[[CAST_MEMREF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<200x100xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CST:.*]] = arith.constant dense<1.100000e+01> : vector<f32>
// CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<f32> to vector<1xf32>
// CHECK: %[[REF:.*]] = llvm.extractvalue %[[CAST_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[MUL:.*]] = llvm.mul %[[I]], %[[C100]] : i64
// CHECK: %[[ADD:.*]] = llvm.add %[[MUL]], %[[J]] : i64
// CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[REF]][%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store %[[VAL]], %[[ADDR]] {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr
// CHECK: return

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
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi32>) -> vector<3x!llvm.ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: return %[[G]] : vector<3xf32>

// -----

func.func @gather_scalable(%arg0: memref<?xf32>, %arg1: vector<[3]xi32>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xf32>) -> vector<[3]xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<[3]xi32>, vector<[3]xi1>, vector<[3]xf32> into vector<[3]xf32>
  return %1 : vector<[3]xf32>
}

// CHECK-LABEL: func @gather_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi32>) -> vector<[3]x!llvm.ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<[3]x!llvm.ptr>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: return %[[G]] : vector<[3]xf32>

// -----

func.func @gather_global_memory(%arg0: memref<?xf32, 1>, %arg1: vector<3xi32>, %arg2: vector<3xi1>, %arg3: vector<3xf32>) -> vector<3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32, 1>, vector<3xi32>, vector<3xi1>, vector<3xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// CHECK-LABEL: func @gather_global_memory
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, vector<3xi32>) -> vector<3x!llvm.ptr<1>>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<3x!llvm.ptr<1>>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// CHECK: return %[[G]] : vector<3xf32>

// -----

func.func @gather_global_memory_scalable(%arg0: memref<?xf32, 1>, %arg1: vector<[3]xi32>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xf32>) -> vector<[3]xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32, 1>, vector<[3]xi32>, vector<[3]xi1>, vector<[3]xf32> into vector<[3]xf32>
  return %1 : vector<[3]xf32>
}

// CHECK-LABEL: func @gather_global_memory_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, vector<[3]xi32>) -> vector<[3]x!llvm.ptr<1>>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<[3]x!llvm.ptr<1>>, vector<[3]xi1>, vector<[3]xf32>) -> vector<[3]xf32>
// CHECK: return %[[G]] : vector<[3]xf32>

// -----


func.func @gather_index(%arg0: memref<?xindex>, %arg1: vector<3xindex>, %arg2: vector<3xi1>, %arg3: vector<3xindex>) -> vector<3xindex> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<3xindex>, vector<3xi1>, vector<3xindex> into vector<3xindex>
  return %1 : vector<3xindex>
}

// CHECK-LABEL: func @gather_index
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi64>) -> vector<3x!llvm.ptr>, i64
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xi64>) -> vector<3xi64>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[G]] : vector<3xi64> to vector<3xindex>

// -----

func.func @gather_index_scalable(%arg0: memref<?xindex>, %arg1: vector<[3]xindex>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xindex>) -> vector<[3]xindex> {
  %0 = arith.constant 0: index
  %1 = vector.gather %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<[3]xindex>, vector<[3]xi1>, vector<[3]xindex> into vector<[3]xindex>
  return %1 : vector<[3]xindex>
}

// CHECK-LABEL: func @gather_index_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi64>) -> vector<[3]x!llvm.ptr>, i64
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (vector<[3]x!llvm.ptr>, vector<[3]xi1>, vector<[3]xi64>) -> vector<[3]xi64>
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[G]] : vector<[3]xi64> to vector<[3]xindex>

// -----

func.func @gather_1d_from_2d(%arg0: memref<4x4xf32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.gather %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x4xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: func @gather_1d_from_2d
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<4xi32>) -> vector<4x!llvm.ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<4x!llvm.ptr>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
// CHECK: return %[[G]] : vector<4xf32>

// -----

func.func @gather_1d_from_2d_scalable(%arg0: memref<4x?xf32>, %arg1: vector<[4]xi32>, %arg2: vector<[4]xi1>, %arg3: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = arith.constant 3 : index
  %1 = vector.gather %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x?xf32>, vector<[4]xi32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
  return %1 : vector<[4]xf32>
}

// CHECK-LABEL: func @gather_1d_from_2d_scalable
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<[4]xi32>) -> vector<[4]x!llvm.ptr>, f32
// CHECK: %[[G:.*]] = llvm.intr.masked.gather %[[P]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<[4]x!llvm.ptr>, vector<[4]xi1>, vector<[4]xf32>) -> vector<[4]xf32>
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
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi32>) -> vector<3x!llvm.ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<3xf32>, vector<3xi1> into vector<3x!llvm.ptr>

// -----

func.func @scatter_scalable(%arg0: memref<?xf32>, %arg1: vector<[3]xi32>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xf32>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xf32>, vector<[3]xi32>, vector<[3]xi1>, vector<[3]xf32>
  return
}

// CHECK-LABEL: func @scatter_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi32>) -> vector<[3]x!llvm.ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<[3]xf32>, vector<[3]xi1> into vector<[3]x!llvm.ptr>

// -----

func.func @scatter_index(%arg0: memref<?xindex>, %arg1: vector<3xindex>, %arg2: vector<3xi1>, %arg3: vector<3xindex>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<3xindex>, vector<3xi1>, vector<3xindex>
  return
}

// CHECK-LABEL: func @scatter_index
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<3xi64>) -> vector<3x!llvm.ptr>, i64
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 8 : i32} : vector<3xi64>, vector<3xi1> into vector<3x!llvm.ptr>

// -----

func.func @scatter_index_scalable(%arg0: memref<?xindex>, %arg1: vector<[3]xindex>, %arg2: vector<[3]xi1>, %arg3: vector<[3]xindex>) {
  %0 = arith.constant 0: index
  vector.scatter %arg0[%0][%arg1], %arg2, %arg3 : memref<?xindex>, vector<[3]xindex>, vector<[3]xi1>, vector<[3]xindex>
  return
}

// CHECK-LABEL: func @scatter_index_scalable
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, vector<[3]xi64>) -> vector<[3]x!llvm.ptr>, i64
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 8 : i32} : vector<[3]xi64>, vector<[3]xi1> into vector<[3]x!llvm.ptr>

// -----

func.func @scatter_1d_into_2d(%arg0: memref<4x4xf32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: vector<4xf32>) {
  %0 = arith.constant 3 : index
  vector.scatter %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x4xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL: func @scatter_1d_into_2d
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<4xi32>) -> vector<4x!llvm.ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<4xf32>, vector<4xi1> into vector<4x!llvm.ptr>

// -----

func.func @scatter_1d_into_2d_scalable(%arg0: memref<4x?xf32>, %arg1: vector<[4]xi32>, %arg2: vector<[4]xi1>, %arg3: vector<[4]xf32>) {
  %0 = arith.constant 3 : index
  vector.scatter %arg0[%0, %0][%arg1], %arg2, %arg3 : memref<4x?xf32>, vector<[4]xi32>, vector<[4]xi1>, vector<[4]xf32>
  return
}

// CHECK-LABEL: func @scatter_1d_into_2d_scalable
// CHECK: %[[B:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[P:.*]] = llvm.getelementptr %[[B]][%{{.*}}] : (!llvm.ptr, vector<[4]xi32>) -> vector<[4]x!llvm.ptr>, f32
// CHECK: llvm.intr.masked.scatter %{{.*}}, %[[P]], %{{.*}} {alignment = 4 : i32} : vector<[4]xf32>, vector<[4]xi1> into vector<[4]x!llvm.ptr>

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

// vector.splat is converted to vector.broadcast. Then, vector.broadcast is converted to LLVM.
// CHECK-LABEL: @splat_0d
// CHECK-NOT: splat
// CHECK: return
func.func @splat_0d(%elt: f32) -> (vector<f32>, vector<4xf32>, vector<[4]xf32>) {
  %a = vector.splat %elt : vector<f32>
  %b = vector.splat %elt : vector<4xf32>
  %c = vector.splat %elt : vector<[4]xf32>
  return %a, %b, %c : vector<f32>, vector<4xf32>, vector<[4]xf32>
}

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

//===----------------------------------------------------------------------===//
// vector.from_elements
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @from_elements_1d(
//  CHECK-SAME:     %[[ARG_0:.*]]: f32, %[[ARG_1:.*]]: f32)
//       CHECK:   %[[UNDEF:.*]] = llvm.mlir.poison : vector<3xf32>
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
//       CHECK:   %[[UNDEF:.*]] = llvm.mlir.poison : vector<1xf32>
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
// vector.to_elements
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @to_elements_no_dead_elements
 // CHECK-SAME:     %[[A:.*]]: vector<4xf32>)
 //      CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
 //      CHECK:   %[[ELEM0:.*]] = llvm.extractelement %[[A]][%[[C0]] : i64] : vector<4xf32>
 //      CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
 //      CHECK:   %[[ELEM1:.*]] = llvm.extractelement %[[A]][%[[C1]] : i64] : vector<4xf32>
 //      CHECK:   %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
 //      CHECK:   %[[ELEM2:.*]] = llvm.extractelement %[[A]][%[[C2]] : i64] : vector<4xf32>
 //      CHECK:   %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
 //      CHECK:   %[[ELEM3:.*]] = llvm.extractelement %[[A]][%[[C3]] : i64] : vector<4xf32>
 //      CHECK:   return %[[ELEM0]], %[[ELEM1]], %[[ELEM2]], %[[ELEM3]] : f32, f32, f32, f32
func.func @to_elements_no_dead_elements(%a: vector<4xf32>) -> (f32, f32, f32, f32) {
  %0:4 = vector.to_elements %a : vector<4xf32>
  return %0#0, %0#1, %0#2, %0#3 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func.func @to_elements_dead_elements
 // CHECK-SAME:     %[[A:.*]]: vector<4xf32>)
 //  CHECK-NOT:   llvm.mlir.constant(0 : i64) : i64
 //      CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
 //      CHECK:   %[[ELEM1:.*]] = llvm.extractelement %[[A]][%[[C1]] : i64] : vector<4xf32>
 //  CHECK-NOT:   llvm.mlir.constant(2 : i64) : i64
 //      CHECK:   %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
 //      CHECK:   %[[ELEM3:.*]] = llvm.extractelement %[[A]][%[[C3]] : i64] : vector<4xf32>
 //      CHECK:   return %[[ELEM1]], %[[ELEM3]] : f32, f32
func.func @to_elements_dead_elements(%a: vector<4xf32>) -> (f32, f32) {
  %0:4 = vector.to_elements %a : vector<4xf32>
  return %0#1, %0#3 : f32, f32
}

// -----

//===----------------------------------------------------------------------===//
// vector.step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @step_scalable
// CHECK: %[[STEPVECTOR:.*]] = llvm.intr.stepvector : vector<[4]xi64>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[STEPVECTOR]] : vector<[4]xi64> to vector<[4]xindex>
// CHECK: return %[[CAST]] : vector<[4]xindex>
func.func @step_scalable() -> vector<[4]xindex> {
  %0 = vector.step : vector<[4]xindex>
  return %0 : vector<[4]xindex>
}
