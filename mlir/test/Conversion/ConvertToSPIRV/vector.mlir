// RUN: mlir-opt -convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

// CHECK-LABEL: @extract
//  CHECK-SAME: %[[ARG:.+]]: vector<2xf32>
//       CHECK:   spirv.CompositeExtract %[[ARG]][0 : i32] : vector<2xf32>
//       CHECK:   spirv.CompositeExtract %[[ARG]][1 : i32] : vector<2xf32>
func.func @extract(%arg0 : vector<2xf32>) -> (vector<1xf32>, f32) {
  %0 = "vector.extract"(%arg0) <{static_position = array<i64: 0>}> : (vector<2xf32>) -> vector<1xf32>
  %1 = "vector.extract"(%arg0) <{static_position = array<i64: 1>}> : (vector<2xf32>) -> f32
  return %0, %1: vector<1xf32>, f32
}

// -----

// CHECK-LABEL: @extract_size1_vector
//  CHECK-SAME: %[[ARG0:.+]]: f32
//       CHECK:   spirv.ReturnValue %[[ARG0]] : f32
func.func @extract_size1_vector(%arg0 : vector<1xf32>) -> f32 {
  %0 = vector.extract %arg0[0] : f32 from vector<1xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @insert
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>, %[[S:.*]]: f32
//       CHECK:   spirv.CompositeInsert %[[S]], %[[V]][2 : i32] : f32 into vector<4xf32>
func.func @insert(%arg0 : vector<4xf32>, %arg1: f32) -> vector<4xf32> {
  %1 = vector.insert %arg1, %arg0[2] : f32 into vector<4xf32>
  return %1: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_index_vector
//       CHECK:   spirv.CompositeInsert %{{.+}}, %{{.+}}[2 : i32] : i32 into vector<4xi32>
func.func @insert_index_vector(%arg0 : vector<4xindex>, %arg1: index) -> vector<4xindex> {
  %1 = vector.insert %arg1, %arg0[2] : index into vector<4xindex>
  return %1: vector<4xindex>
}

// -----

// CHECK-LABEL: @insert_size1_vector
//  CHECK-SAME: %[[V:.*]]: f32, %[[S:.*]]: f32
//       CHECK:   spirv.ReturnValue %[[S]] : f32
func.func @insert_size1_vector(%arg0 : vector<1xf32>, %arg1: f32) -> vector<1xf32> {
  %1 = vector.insert %arg1, %arg0[0] : f32 into vector<1xf32>
  return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL: @extract_element
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>, %[[ID:.*]]: i32
//       CHECK:   spirv.VectorExtractDynamic %[[V]][%[[ID]]] : vector<4xf32>, i32
func.func @extract_element(%arg0 : vector<4xf32>, %id : i32) -> f32 {
  %0 = vector.extractelement %arg0[%id : i32] : vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_cst
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>
//       CHECK:   spirv.CompositeExtract %[[V]][1 : i32] : vector<4xf32>
func.func @extract_element_cst(%arg0 : vector<4xf32>) -> f32 {
  %idx = arith.constant 1 : i32
  %0 = vector.extractelement %arg0[%idx : i32] : vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_index
func.func @extract_element_index(%arg0 : vector<4xf32>, %id : index) -> f32 {
  // CHECK: spirv.VectorExtractDynamic
  %0 = vector.extractelement %arg0[%id : index] : vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_size1_vector
//  CHECK-SAME:(%[[S:.+]]: f32,
func.func @extract_element_size1_vector(%arg0 : f32, %i: index) -> f32 {
  %bcast = vector.broadcast %arg0 : f32 to vector<1xf32>
  %0 = vector.extractelement %bcast[%i : index] : vector<1xf32>
  // CHECK: spirv.ReturnValue %[[S]]
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_0d_vector
//  CHECK-SAME: (%[[S:.+]]: f32)
func.func @extract_element_0d_vector(%arg0 : f32) -> f32 {
  %bcast = vector.broadcast %arg0 : f32 to vector<f32>
  %0 = vector.extractelement %bcast[] : vector<f32>
  // CHECK: spirv.ReturnValue %[[S]]
  return %0: f32
}

// -----

// CHECK-LABEL: @insert_element
//  CHECK-SAME: %[[VAL:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[ID:.*]]: i32
//       CHECK:   spirv.VectorInsertDynamic %[[VAL]], %[[V]][%[[ID]]] : vector<4xf32>, i32
func.func @insert_element(%val: f32, %arg0 : vector<4xf32>, %id : i32) -> vector<4xf32> {
  %0 = vector.insertelement %val, %arg0[%id : i32] : vector<4xf32>
  return %0: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_element_cst
//  CHECK-SAME: %[[VAL:.*]]: f32, %[[V:.*]]: vector<4xf32>
//       CHECK:   spirv.CompositeInsert %[[VAL]], %[[V]][2 : i32] : f32 into vector<4xf32>
func.func @insert_element_cst(%val: f32, %arg0 : vector<4xf32>) -> vector<4xf32> {
  %idx = arith.constant 2 : i32
  %0 = vector.insertelement %val, %arg0[%idx : i32] : vector<4xf32>
  return %0: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_element_index
func.func @insert_element_index(%val: f32, %arg0 : vector<4xf32>, %id : index) -> vector<4xf32> {
  // CHECK: spirv.VectorInsertDynamic
  %0 = vector.insertelement %val, %arg0[%id : index] : vector<4xf32>
  return %0: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_element_size1_vector
//  CHECK-SAME: (%[[S:[a-z0-9]+]]: f32
func.func @insert_element_size1_vector(%scalar: f32, %vector : vector<1xf32>, %i: index) -> vector<1xf32> {
  %0 = vector.insertelement %scalar, %vector[%i : index] : vector<1xf32>
  // CHECK: spirv.ReturnValue %[[S]]
  return %0: vector<1xf32>
}

// -----

// CHECK-LABEL: @insert_element_0d_vector
//  CHECK-SAME: (%[[S:[a-z0-9]+]]: f32
func.func @insert_element_0d_vector(%scalar: f32, %vector : vector<f32>) -> vector<f32> {
  %0 = vector.insertelement %scalar, %vector[] : vector<f32>
  // CHECK: spirv.ReturnValue %[[S]]
  return %0: vector<f32>
}

// -----

// CHECK-LABEL: @insert_size1_vector
//  CHECK-SAME: %[[SUB:.*]]: f32, %[[FULL:.*]]: vector<3xf32>
//       CHECK:   %[[RET:.*]] = spirv.CompositeInsert %[[SUB]], %[[FULL]][2 : i32] : f32 into vector<3xf32>
//       CHECK:   spirv.ReturnValue %[[RET]] : vector<3xf32>
func.func @insert_size1_vector(%arg0 : vector<1xf32>, %arg1: vector<3xf32>) -> vector<3xf32> {
  %1 = vector.insert_strided_slice %arg0, %arg1 {offsets = [2], strides = [1]} : vector<1xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// -----

// CHECK-LABEL: @fma
//  CHECK-SAME: %[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: vector<4xf32>
//       CHECK:   spirv.GL.Fma %[[A]], %[[B]], %[[C]] : vector<4xf32>
func.func @fma(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.fma %a, %b, %c: vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @fma_size1_vector
//       CHECK:   spirv.GL.Fma %{{.+}} : f32
func.func @fma_size1_vector(%a: vector<1xf32>, %b: vector<1xf32>, %c: vector<1xf32>) -> vector<1xf32> {
  %0 = vector.fma %a, %b, %c: vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// CHECK-LABEL: func @splat
//  CHECK-SAME: (%[[A:.+]]: f32)
//       CHECK:   %[[VAL:.+]] = spirv.CompositeConstruct %[[A]], %[[A]], %[[A]], %[[A]]
//       CHECK:   spirv.ReturnValue %[[VAL]] : vector<4xf32>
func.func @splat(%f : f32) -> vector<4xf32> {
  %splat = vector.splat %f : vector<4xf32>
  return %splat : vector<4xf32>
}

// -----

// CHECK-LABEL: func @splat_size1_vector
//  CHECK-SAME: (%[[A:.+]]: f32)
//       CHECK:   spirv.ReturnValue %[[A]] : f32
func.func @splat_size1_vector(%f : f32) -> vector<1xf32> {
  %splat = vector.splat %f : vector<1xf32>
  return %splat : vector<1xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32
//       CHECK:    spirv.CompositeConstruct %[[ARG0]], %[[ARG1]], %[[ARG1]], %[[ARG0]] : (f32, f32, f32, f32) -> vector<4xf32>
func.func @shuffle(%v0 : vector<1xf32>, %v1: vector<1xf32>) -> vector<4xf32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 1, 0] : vector<1xf32>, vector<1xf32>
  return %shuffle : vector<4xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[V0:.+]]: vector<3xf32>, %[[V1:.+]]: vector<3xf32>
//       CHECK:    spirv.VectorShuffle [3 : i32, 2 : i32, 5 : i32, 1 : i32] %[[V0]], %[[V1]] : vector<3xf32>, vector<3xf32> -> vector<4xf32>
func.func @shuffle(%v0 : vector<3xf32>, %v1: vector<3xf32>) -> vector<4xf32> {
  %shuffle = vector.shuffle %v0, %v1 [3, 2, 5, 1] : vector<3xf32>, vector<3xf32>
  return %shuffle : vector<4xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: i32, %[[ARG1:.+]]: vector<3xi32>
//       CHECK:    %[[S1:.+]] = spirv.CompositeExtract %[[ARG1]][1 : i32] : vector<3xi32>
//       CHECK:    %[[S2:.+]] = spirv.CompositeExtract %[[ARG1]][2 : i32] : vector<3xi32>
//       CHECK:    %[[RES:.+]] = spirv.CompositeConstruct %[[ARG0]], %[[S1]], %[[S2]] : (i32, i32, i32) -> vector<3xi32>
//       CHECK:    spirv.ReturnValue %[[RES]]
func.func @shuffle(%v0 : vector<1xi32>, %v1: vector<3xi32>) -> vector<3xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 2, 3] : vector<1xi32>, vector<3xi32>
  return %shuffle : vector<3xi32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: i32, %[[ARG1:.+]]: i32
//       CHECK:    %[[RES:.+]] = spirv.CompositeConstruct %[[ARG0]], %[[ARG1]] : (i32, i32) -> vector<2xi32>
//       CHECK:    spirv.ReturnValue %[[RES]]
func.func @shuffle(%v0 : vector<1xi32>, %v1: vector<1xi32>) -> vector<2xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1] : vector<1xi32>, vector<1xi32>
  return %shuffle : vector<2xi32>
}

// -----

// CHECK-LABEL: func @interleave
//  CHECK-SAME: (%[[ARG0:.+]]: vector<2xf32>, %[[ARG1:.+]]: vector<2xf32>)
//       CHECK: %[[SHUFFLE:.*]] = spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32] %[[ARG0]], %[[ARG1]] : vector<2xf32>, vector<2xf32> -> vector<4xf32>
//       CHECK: spirv.ReturnValue %[[SHUFFLE]]
func.func @interleave(%a: vector<2xf32>, %b: vector<2xf32>) -> vector<4xf32> {
  %0 = vector.interleave %a, %b : vector<2xf32> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @interleave_size1
// CHECK-SAME: (%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32)
//       CHECK: %[[RES:.*]] = spirv.CompositeConstruct %[[ARG0]], %[[ARG1]] : (f32, f32) -> vector<2xf32>
//       CHECK: spirv.ReturnValue %[[RES]]
func.func @interleave_size1(%a: vector<1xf32>, %b: vector<1xf32>) -> vector<2xf32> {
  %0 = vector.interleave %a, %b : vector<1xf32> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @reduction_add
//  CHECK-SAME: (%[[V:.+]]: vector<4xi32>)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<4xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<4xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<4xi32>
//       CHECK:   %[[S3:.+]] = spirv.CompositeExtract %[[V]][3 : i32] : vector<4xi32>
//       CHECK:   %[[ADD0:.+]] = spirv.IAdd %[[S0]], %[[S1]]
//       CHECK:   %[[ADD1:.+]] = spirv.IAdd %[[ADD0]], %[[S2]]
//       CHECK:   %[[ADD2:.+]] = spirv.IAdd %[[ADD1]], %[[S3]]
//       CHECK:   spirv.ReturnValue %[[ADD2]]
func.func @reduction_add(%v : vector<4xi32>) -> i32 {
  %reduce = vector.reduction <add>, %v : vector<4xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: func @reduction_addf_one_elem
//  CHECK-SAME:  (%[[ARG0:.+]]: f32)
//  CHECK:       spirv.ReturnValue %[[ARG0]] : f32
func.func @reduction_addf_one_elem(%arg0: vector<1xf32>) -> f32 {
  %red = vector.reduction <add>, %arg0 : vector<1xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_addf_one_elem_acc
//  CHECK-SAME:  (%[[ARG0:.+]]: f32, %[[ACC:.+]]: f32)
//  CHECK:       %[[RES:.+]] = spirv.FAdd %[[ACC]], %[[ARG0]] : f32
//  CHECK:       spirv.ReturnValue %[[RES]] : f32
func.func @reduction_addf_one_elem_acc(%arg0: vector<1xf32>, %acc: f32) -> f32 {
  %red = vector.reduction <add>, %arg0, %acc : vector<1xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_mul
//  CHECK-SAME: (%[[V:.+]]: vector<3xf32>, %[[S:.+]]: f32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xf32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xf32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xf32>
//       CHECK:   %[[MUL0:.+]] = spirv.FMul %[[S0]], %[[S1]]
//       CHECK:   %[[MUL1:.+]] = spirv.FMul %[[MUL0]], %[[S2]]
//       CHECK:   %[[MUL2:.+]] = spirv.FMul %[[MUL1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MUL2]]
func.func @reduction_mul(%v : vector<3xf32>, %s: f32) -> f32 {
  %reduce = vector.reduction <mul>, %v, %s : vector<3xf32> into f32
  return %reduce : f32
}

// -----

// CHECK-LABEL: func @reduction_maximumf
//  CHECK-SAME: (%[[V:.+]]: vector<3xf32>, %[[S:.+]]: f32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xf32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xf32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xf32>
//       CHECK:   %[[MAX0:.+]] = spirv.GL.FMax %[[S0]], %[[S1]]
//       CHECK:   %[[MAX1:.+]] = spirv.GL.FMax %[[MAX0]], %[[S2]]
//       CHECK:   %[[MAX2:.+]] = spirv.GL.FMax %[[MAX1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MAX2]]
func.func @reduction_maximumf(%v : vector<3xf32>, %s: f32) -> f32 {
  %reduce = vector.reduction <maximumf>, %v, %s : vector<3xf32> into f32
  return %reduce : f32
}

// -----

// CHECK-LABEL: func @reduction_minimumf
//  CHECK-SAME: (%[[V:.+]]: vector<3xf32>, %[[S:.+]]: f32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xf32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xf32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xf32>
//       CHECK:   %[[MIN0:.+]] = spirv.GL.FMin %[[S0]], %[[S1]]
//       CHECK:   %[[MIN1:.+]] = spirv.GL.FMin %[[MIN0]], %[[S2]]
//       CHECK:   %[[MIN2:.+]] = spirv.GL.FMin %[[MIN1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MIN2]]
func.func @reduction_minimumf(%v : vector<3xf32>, %s: f32) -> f32 {
  %reduce = vector.reduction <minimumf>, %v, %s : vector<3xf32> into f32
  return %reduce : f32
}

// -----

// CHECK-LABEL: func @reduction_maxsi
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MAX0:.+]] = spirv.GL.SMax %[[S0]], %[[S1]]
//       CHECK:   %[[MAX1:.+]] = spirv.GL.SMax %[[MAX0]], %[[S2]]
//       CHECK:   %[[MAX2:.+]] = spirv.GL.SMax %[[MAX1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MAX2]]
func.func @reduction_maxsi(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <maxsi>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: func @reduction_minsi
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MIN0:.+]] = spirv.GL.SMin %[[S0]], %[[S1]]
//       CHECK:   %[[MIN1:.+]] = spirv.GL.SMin %[[MIN0]], %[[S2]]
//       CHECK:   %[[MIN2:.+]] = spirv.GL.SMin %[[MIN1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MIN2]]
func.func @reduction_minsi(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <minsi>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: func @reduction_maxui
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MAX0:.+]] = spirv.GL.UMax %[[S0]], %[[S1]]
//       CHECK:   %[[MAX1:.+]] = spirv.GL.UMax %[[MAX0]], %[[S2]]
//       CHECK:   %[[MAX2:.+]] = spirv.GL.UMax %[[MAX1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MAX2]]
func.func @reduction_maxui(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <maxui>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: func @reduction_minui
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MIN0:.+]] = spirv.GL.UMin %[[S0]], %[[S1]]
//       CHECK:   %[[MIN1:.+]] = spirv.GL.UMin %[[MIN0]], %[[S2]]
//       CHECK:   %[[MIN2:.+]] = spirv.GL.UMin %[[MIN1]], %[[S]]
//       CHECK:   spirv.ReturnValue %[[MIN2]]
func.func @reduction_minui(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <minui>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: @shape_cast_same_type
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2xf32>)
//       CHECK:   spirv.ReturnValue %[[ARG0]]
func.func @shape_cast_same_type(%arg0 : vector<2xf32>) -> vector<2xf32> {
  %1 = vector.shape_cast %arg0 : vector<2xf32> to vector<2xf32>
  return %arg0 : vector<2xf32>
}

// -----

// CHECK-LABEL: @shape_cast_size1_vector
//  CHECK-SAME: (%[[ARG0:.*]]: f32)
//       CHECK:   spirv.ReturnValue %[[ARG0]]
func.func @shape_cast_size1_vector(%arg0 : vector<f32>) -> vector<1xf32> {
  %1 = vector.shape_cast %arg0 : vector<f32> to vector<1xf32>
  return %1 : vector<1xf32>
}
