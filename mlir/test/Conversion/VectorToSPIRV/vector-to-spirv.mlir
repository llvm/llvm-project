// RUN: mlir-opt -split-input-file -convert-vector-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes { spv.target_env = #spv.target_env<#spv.vce<v1.0, [Float16], []>, #spv.resource_limits<>> } {

// CHECK-LABEL: @bitcast
//  CHECK-SAME: %[[ARG0:.+]]: vector<2xf32>, %[[ARG1:.+]]: vector<2xf16>
//       CHECK:   spv.Bitcast %[[ARG0]] : vector<2xf32> to vector<4xf16>
//       CHECK:   spv.Bitcast %[[ARG1]] : vector<2xf16> to f32
func.func @bitcast(%arg0 : vector<2xf32>, %arg1: vector<2xf16>) -> (vector<4xf16>, vector<1xf32>) {
  %0 = vector.bitcast %arg0 : vector<2xf32> to vector<4xf16>
  %1 = vector.bitcast %arg1 : vector<2xf16> to vector<1xf32>
  return %0, %1: vector<4xf16>, vector<1xf32>
}

} // end module

// -----

// CHECK-LABEL: @broadcast
//  CHECK-SAME: %[[A:.*]]: f32
//       CHECK:   spv.CompositeConstruct %[[A]], %[[A]], %[[A]], %[[A]] : vector<4xf32>
//       CHECK:   spv.CompositeConstruct %[[A]], %[[A]] : vector<2xf32>
func.func @broadcast(%arg0 : f32) -> (vector<4xf32>, vector<2xf32>) {
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %1 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0, %1: vector<4xf32>, vector<2xf32>
}

// -----

// CHECK-LABEL: @extract
//  CHECK-SAME: %[[ARG:.+]]: vector<2xf32>
//       CHECK:   spv.CompositeExtract %[[ARG]][0 : i32] : vector<2xf32>
//       CHECK:   spv.CompositeExtract %[[ARG]][1 : i32] : vector<2xf32>
func.func @extract(%arg0 : vector<2xf32>) -> (vector<1xf32>, f32) {
  %0 = "vector.extract"(%arg0) {position = [0]} : (vector<2xf32>) -> vector<1xf32>
  %1 = "vector.extract"(%arg0) {position = [1]} : (vector<2xf32>) -> f32
  return %0, %1: vector<1xf32>, f32
}

// -----

// CHECK-LABEL: @extract_size1_vector
//  CHECK-SAME: %[[ARG0:.+]]: vector<1xf32>
//       CHECK:   %[[R:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//       CHECK:   return %[[R]]
func.func @extract_size1_vector(%arg0 : vector<1xf32>) -> f32 {
  %0 = vector.extract %arg0[0] : vector<1xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @insert
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>, %[[S:.*]]: f32
//       CHECK:   spv.CompositeInsert %[[S]], %[[V]][2 : i32] : f32 into vector<4xf32>
func.func @insert(%arg0 : vector<4xf32>, %arg1: f32) -> vector<4xf32> {
  %1 = vector.insert %arg1, %arg0[2] : f32 into vector<4xf32>
  return %1: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_size1_vector
//  CHECK-SAME: %[[V:.*]]: vector<1xf32>, %[[S:.*]]: f32
//       CHECK:   %[[R:.+]] = builtin.unrealized_conversion_cast %[[S]]
//       CHECK:   return %[[R]]
func.func @insert_size1_vector(%arg0 : vector<1xf32>, %arg1: f32) -> vector<1xf32> {
  %1 = vector.insert %arg1, %arg0[0] : f32 into vector<1xf32>
  return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL: @extract_element
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>, %[[ID:.*]]: i32
//       CHECK:   spv.VectorExtractDynamic %[[V]][%[[ID]]] : vector<4xf32>, i32
func.func @extract_element(%arg0 : vector<4xf32>, %id : i32) -> f32 {
  %0 = vector.extractelement %arg0[%id : i32] : vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_index
func.func @extract_element_index(%arg0 : vector<4xf32>, %id : index) -> f32 {
  // CHECK: vector.extractelement
  %0 = vector.extractelement %arg0[%id : index] : vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_size5_vector
func.func @extract_element_size5_vector(%arg0 : vector<5xf32>, %id : i32) -> f32 {
  // CHECK: vector.extractelement
  %0 = vector.extractelement %arg0[%id : i32] : vector<5xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_strided_slice
//  CHECK-SAME: %[[ARG:.+]]: vector<4xf32>
//       CHECK:   spv.VectorShuffle [1 : i32, 2 : i32] %[[ARG]] : vector<4xf32>, %[[ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   spv.CompositeExtract %[[ARG]][1 : i32] : vector<4xf32>
func.func @extract_strided_slice(%arg0: vector<4xf32>) -> (vector<2xf32>, vector<1xf32>) {
  %0 = vector.extract_strided_slice %arg0 {offsets = [1], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
  %1 = vector.extract_strided_slice %arg0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  return %0, %1 : vector<2xf32>, vector<1xf32>
}

// -----

// CHECK-LABEL: @insert_element
//  CHECK-SAME: %[[VAL:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[ID:.*]]: i32
//       CHECK:   spv.VectorInsertDynamic %[[VAL]], %[[V]][%[[ID]]] : vector<4xf32>, i32
func.func @insert_element(%val: f32, %arg0 : vector<4xf32>, %id : i32) -> vector<4xf32> {
  %0 = vector.insertelement %val, %arg0[%id : i32] : vector<4xf32>
  return %0: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_element_index
func.func @insert_element_index(%val: f32, %arg0 : vector<4xf32>, %id : index) -> vector<4xf32> {
  // CHECK: vector.insertelement
  %0 = vector.insertelement %val, %arg0[%id : index] : vector<4xf32>
  return %0: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_element_size5_vector
func.func @insert_element_size5_vector(%val: f32, %arg0 : vector<5xf32>, %id : i32) -> vector<5xf32> {
  // CHECK: vector.insertelement
  %0 = vector.insertelement %val, %arg0[%id : i32] : vector<5xf32>
  return %0 : vector<5xf32>
}

// -----

// CHECK-LABEL: @insert_strided_slice
//  CHECK-SAME: %[[PART:.+]]: vector<2xf32>, %[[ALL:.+]]: vector<4xf32>
//       CHECK:   spv.VectorShuffle [0 : i32, 4 : i32, 5 : i32, 3 : i32] %[[ALL]] : vector<4xf32>, %[[PART]] : vector<2xf32> -> vector<4xf32>
func.func @insert_strided_slice(%arg0: vector<2xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1], strides = [1]} : vector<2xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_size1_vector
//  CHECK-SAME: %[[SUB:.*]]: vector<1xf32>, %[[FULL:.*]]: vector<3xf32>
//       CHECK:   %[[S:.+]] = builtin.unrealized_conversion_cast %[[SUB]]
//       CHECK:   spv.CompositeInsert %[[S]], %[[FULL]][2 : i32] : f32 into vector<3xf32>
func.func @insert_size1_vector(%arg0 : vector<1xf32>, %arg1: vector<3xf32>) -> vector<3xf32> {
  %1 = vector.insert_strided_slice %arg0, %arg1 {offsets = [2], strides = [1]} : vector<1xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// -----

// CHECK-LABEL: @fma
//  CHECK-SAME: %[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: vector<4xf32>
//       CHECK:   spv.GL.Fma %[[A]], %[[B]], %[[C]] : vector<4xf32>
func.func @fma(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.fma %a, %b, %c: vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @fma_size1_vector
//       CHECK:   spv.GL.Fma %{{.+}} : f32
func.func @fma_size1_vector(%a: vector<1xf32>, %b: vector<1xf32>, %c: vector<1xf32>) -> vector<1xf32> {
  %0 = vector.fma %a, %b, %c: vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// CHECK-LABEL: func @splat
//  CHECK-SAME: (%[[A:.+]]: f32)
//       CHECK:   %[[VAL:.+]] = spv.CompositeConstruct %[[A]], %[[A]], %[[A]], %[[A]] : vector<4xf32>
//       CHECK:   return %[[VAL]]
func.func @splat(%f : f32) -> vector<4xf32> {
  %splat = vector.splat %f : vector<4xf32>
  return %splat : vector<4xf32>
}

// -----

// CHECK-LABEL: func @splat_size1_vector
//  CHECK-SAME: (%[[A:.+]]: f32)
//       CHECK:   %[[VAL:.+]] = builtin.unrealized_conversion_cast %[[A]]
//       CHECK:   return %[[VAL]]
func.func @splat_size1_vector(%f : f32) -> vector<1xf32> {
  %splat = vector.splat %f : vector<1xf32>
  return %splat : vector<1xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<1xf32>, %[[ARG1:.+]]: vector<1xf32>
//       CHECK:    %[[V0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//       CHECK:    %[[V1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]]
//       CHECK:    spv.CompositeConstruct %[[V0]], %[[V1]], %[[V1]], %[[V0]] : vector<4xf32>
func.func @shuffle(%v0 : vector<1xf32>, %v1: vector<1xf32>) -> vector<4xf32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 1, 0] : vector<1xf32>, vector<1xf32>
  return %shuffle : vector<4xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[V0:.+]]: vector<3xf32>, %[[V1:.+]]: vector<3xf32>
//       CHECK:    spv.VectorShuffle [3 : i32, 2 : i32, 5 : i32, 1 : i32] %[[V0]] : vector<3xf32>, %[[V1]] : vector<3xf32> -> vector<4xf32>
func.func @shuffle(%v0 : vector<3xf32>, %v1: vector<3xf32>) -> vector<4xf32> {
  %shuffle = vector.shuffle %v0, %v1 [3, 2, 5, 1] : vector<3xf32>, vector<3xf32>
  return %shuffle : vector<4xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
func.func @shuffle(%v0 : vector<2x16xf32>, %v1: vector<1x16xf32>) -> vector<3x16xf32> {
  // CHECK: vector.shuffle
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 2] : vector<2x16xf32>, vector<1x16xf32>
  return %shuffle : vector<3x16xf32>
}

// -----

// CHECK-LABEL: func @reduction
//  CHECK-SAME: (%[[V:.+]]: vector<4xi32>)
//       CHECK:   %[[S0:.+]] = spv.CompositeExtract %[[V]][0 : i32] : vector<4xi32>
//       CHECK:   %[[S1:.+]] = spv.CompositeExtract %[[V]][1 : i32] : vector<4xi32>
//       CHECK:   %[[S2:.+]] = spv.CompositeExtract %[[V]][2 : i32] : vector<4xi32>
//       CHECK:   %[[S3:.+]] = spv.CompositeExtract %[[V]][3 : i32] : vector<4xi32>
//       CHECK:   %[[ADD0:.+]] = spv.IAdd %[[S0]], %[[S1]]
//       CHECK:   %[[ADD1:.+]] = spv.IAdd %[[ADD0]], %[[S2]]
//       CHECK:   %[[ADD2:.+]] = spv.IAdd %[[ADD1]], %[[S3]]
//       CHECK:   return %[[ADD2]]
func.func @reduction(%v : vector<4xi32>) -> i32 {
  %reduce = vector.reduction <add>, %v : vector<4xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: func @reduction
//  CHECK-SAME: (%[[V:.+]]: vector<3xf32>, %[[S:.+]]: f32)
//       CHECK:   %[[S0:.+]] = spv.CompositeExtract %[[V]][0 : i32] : vector<3xf32>
//       CHECK:   %[[S1:.+]] = spv.CompositeExtract %[[V]][1 : i32] : vector<3xf32>
//       CHECK:   %[[S2:.+]] = spv.CompositeExtract %[[V]][2 : i32] : vector<3xf32>
//       CHECK:   %[[ADD0:.+]] = spv.FMul %[[S0]], %[[S1]]
//       CHECK:   %[[ADD1:.+]] = spv.FMul %[[ADD0]], %[[S2]]
//       CHECK:   %[[ADD2:.+]] = spv.FMul %[[ADD1]], %[[S]]
//       CHECK:   return %[[ADD2]]
func.func @reduction(%v : vector<3xf32>, %s: f32) -> f32 {
  %reduce = vector.reduction <mul>, %v, %s : vector<3xf32> into f32
  return %reduce : f32
}
