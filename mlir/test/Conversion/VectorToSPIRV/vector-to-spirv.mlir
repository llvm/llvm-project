// RUN: mlir-opt -split-input-file -convert-vector-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes { spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float16], []>, #spirv.resource_limits<>> } {

// CHECK-LABEL: @bitcast
//  CHECK-SAME: %[[ARG0:.+]]: vector<2xf32>, %[[ARG1:.+]]: vector<2xf16>
//       CHECK:   spirv.Bitcast %[[ARG0]] : vector<2xf32> to vector<4xf16>
//       CHECK:   spirv.Bitcast %[[ARG1]] : vector<2xf16> to f32
func.func @bitcast(%arg0 : vector<2xf32>, %arg1: vector<2xf16>) -> (vector<4xf16>, vector<1xf32>) {
  %0 = vector.bitcast %arg0 : vector<2xf32> to vector<4xf16>
  %1 = vector.bitcast %arg1 : vector<2xf16> to vector<1xf32>
  return %0, %1: vector<4xf16>, vector<1xf32>
}

} // end module

// -----

// Check that without the proper capability we fail the pattern application
// to avoid generating invalid ops.

module attributes { spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>> } {

// CHECK-LABEL: @bitcast
func.func @bitcast(%arg0 : vector<2xf32>, %arg1: vector<2xf16>) -> (vector<4xf16>, vector<1xf32>) {
  // CHECK-COUNT-2: vector.bitcast
  %0 = vector.bitcast %arg0 : vector<2xf32> to vector<4xf16>
  %1 = vector.bitcast %arg1 : vector<2xf16> to vector<1xf32>
  return %0, %1: vector<4xf16>, vector<1xf32>
}

} // end module

// -----

module attributes { spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>> } {

// CHECK-LABEL: @cl_fma
//  CHECK-SAME: %[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: vector<4xf32>
//       CHECK:   spirv.CL.fma %[[A]], %[[B]], %[[C]] : vector<4xf32>
func.func @cl_fma(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.fma %a, %b, %c: vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @cl_fma_size1_vector
//       CHECK:   spirv.CL.fma %{{.+}} : f32
func.func @cl_fma_size1_vector(%a: vector<1xf32>, %b: vector<1xf32>, %c: vector<1xf32>) -> vector<1xf32> {
  %0 = vector.fma %a, %b, %c: vector<1xf32>
  return %0 : vector<1xf32>
}

// CHECK-LABEL: func @cl_reduction_maximumf
//  CHECK-SAME: (%[[V:.+]]: vector<3xf32>, %[[S:.+]]: f32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xf32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xf32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xf32>
//       CHECK:   %[[MAX0:.+]] = spirv.CL.fmax %[[S0]], %[[S1]]
//       CHECK:   %[[MAX1:.+]] = spirv.CL.fmax %[[MAX0]], %[[S2]]
//       CHECK:   %[[MAX2:.+]] = spirv.CL.fmax %[[MAX1]], %[[S]]
//       CHECK:   return %[[MAX2]]
func.func @cl_reduction_maximumf(%v : vector<3xf32>, %s: f32) -> f32 {
  %reduce = vector.reduction <maximumf>, %v, %s : vector<3xf32> into f32
  return %reduce : f32
}

// CHECK-LABEL: func @cl_reduction_minimumf
//  CHECK-SAME: (%[[V:.+]]: vector<3xf32>, %[[S:.+]]: f32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xf32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xf32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xf32>
//       CHECK:   %[[MIN0:.+]] = spirv.CL.fmin %[[S0]], %[[S1]]
//       CHECK:   %[[MIN1:.+]] = spirv.CL.fmin %[[MIN0]], %[[S2]]
//       CHECK:   %[[MIN2:.+]] = spirv.CL.fmin %[[MIN1]], %[[S]]
//       CHECK:   return %[[MIN2]]
func.func @cl_reduction_minimumf(%v : vector<3xf32>, %s: f32) -> f32 {
  %reduce = vector.reduction <minimumf>, %v, %s : vector<3xf32> into f32
  return %reduce : f32
}

// CHECK-LABEL: func @cl_reduction_maxsi
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MAX0:.+]] = spirv.CL.s_max %[[S0]], %[[S1]]
//       CHECK:   %[[MAX1:.+]] = spirv.CL.s_max %[[MAX0]], %[[S2]]
//       CHECK:   %[[MAX2:.+]] = spirv.CL.s_max %[[MAX1]], %[[S]]
//       CHECK:   return %[[MAX2]]
func.func @cl_reduction_maxsi(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <maxsi>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// CHECK-LABEL: func @cl_reduction_minsi
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MIN0:.+]] = spirv.CL.s_min %[[S0]], %[[S1]]
//       CHECK:   %[[MIN1:.+]] = spirv.CL.s_min %[[MIN0]], %[[S2]]
//       CHECK:   %[[MIN2:.+]] = spirv.CL.s_min %[[MIN1]], %[[S]]
//       CHECK:   return %[[MIN2]]
func.func @cl_reduction_minsi(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <minsi>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// CHECK-LABEL: func @cl_reduction_maxui
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MAX0:.+]] = spirv.CL.u_max %[[S0]], %[[S1]]
//       CHECK:   %[[MAX1:.+]] = spirv.CL.u_max %[[MAX0]], %[[S2]]
//       CHECK:   %[[MAX2:.+]] = spirv.CL.u_max %[[MAX1]], %[[S]]
//       CHECK:   return %[[MAX2]]
func.func @cl_reduction_maxui(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <maxui>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// CHECK-LABEL: func @cl_reduction_minui
//  CHECK-SAME: (%[[V:.+]]: vector<3xi32>, %[[S:.+]]: i32)
//       CHECK:   %[[S0:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<3xi32>
//       CHECK:   %[[S1:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<3xi32>
//       CHECK:   %[[S2:.+]] = spirv.CompositeExtract %[[V]][2 : i32] : vector<3xi32>
//       CHECK:   %[[MIN0:.+]] = spirv.CL.u_min %[[S0]], %[[S1]]
//       CHECK:   %[[MIN1:.+]] = spirv.CL.u_min %[[MIN0]], %[[S2]]
//       CHECK:   %[[MIN2:.+]] = spirv.CL.u_min %[[MIN1]], %[[S]]
//       CHECK:   return %[[MIN2]]
func.func @cl_reduction_minui(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <minui>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

} // end module

// -----

// CHECK-LABEL: @broadcast
//  CHECK-SAME: %[[A:.*]]: f32
//       CHECK:   spirv.CompositeConstruct %[[A]], %[[A]], %[[A]], %[[A]]
//       CHECK:   spirv.CompositeConstruct %[[A]], %[[A]]
func.func @broadcast(%arg0 : f32) -> (vector<4xf32>, vector<2xf32>) {
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %1 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0, %1: vector<4xf32>, vector<2xf32>
}

// -----

// CHECK-LABEL: @broadcast_index
//  CHECK-SAME: %[[ARG0:.*]]: index
//       CHECK:   %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : index to i32
//       CHECK:   %[[CONSTRUCT:.*]] = spirv.CompositeConstruct %[[CAST0]], %[[CAST0]], %[[CAST0]], %[[CAST0]] : (i32, i32, i32, i32) -> vector<4xi32>
//       CHECK:   %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[CONSTRUCT]] : vector<4xi32> to vector<4xindex>
//       CHECK:   return %[[CAST1]] : vector<4xindex>
func.func @broadcast_index(%a: index) -> vector<4xindex> {
  %0 = vector.broadcast %a : index to vector<4xindex>
  return %0 : vector<4xindex>
}

// -----

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

func.func @extract_poison_idx(%arg0 : vector<4xf32>) -> f32 {
  // expected-error@+1 {{index -1 out of bounds for 'vector<4xf32>'}}
  %0 = vector.extract %arg0[-1] : f32 from vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_size1_vector
//  CHECK-SAME: %[[ARG0:.+]]: vector<1xf32>
//       CHECK:   %[[R:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//       CHECK:   return %[[R]]
func.func @extract_size1_vector(%arg0 : vector<1xf32>) -> f32 {
  %0 = vector.extract %arg0[0] : f32 from vector<1xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_size1_vector_dynamic
//  CHECK-SAME: %[[ARG0:.+]]: vector<1xf32>
//       CHECK:   %[[R:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//       CHECK:   return %[[R]]
func.func @extract_size1_vector_dynamic(%arg0 : vector<1xf32>, %id : index) -> f32 {
  %0 = vector.extract %arg0[%id] : f32 from vector<1xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_dynamic
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>, %[[ARG1:.*]]: index
//       CHECK:   %[[ID:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i32
//       CHECK:   spirv.VectorExtractDynamic %[[V]][%[[ID]]] : vector<4xf32>, i32
func.func @extract_dynamic(%arg0 : vector<4xf32>, %id : index) -> f32 {
  %0 = vector.extract %arg0[%id] : f32 from vector<4xf32>
  return %0: f32
}

// CHECK-LABEL: @extract_dynamic_cst
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>
//       CHECK:   spirv.CompositeExtract %[[V]][1 : i32] : vector<4xf32>
func.func @extract_dynamic_cst(%arg0 : vector<4xf32>) -> f32 {
  %idx = arith.constant 1 : index
  %0 = vector.extract %arg0[%idx] : f32 from vector<4xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @from_elements_0d
//  CHECK-SAME: %[[ARG0:.+]]: f32
//       CHECK:   %[[RETVAL:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//       CHECK:   return %[[RETVAL]]
func.func @from_elements_0d(%arg0 : f32) -> vector<f32> {
  %0 = vector.from_elements %arg0 : vector<f32>
  return %0: vector<f32>
}

// CHECK-LABEL: @from_elements_1x
//  CHECK-SAME: %[[ARG0:.+]]: f32
//       CHECK:   %[[RETVAL:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//       CHECK:   return %[[RETVAL]]
func.func @from_elements_1x(%arg0 : f32) -> vector<1xf32> {
  %0 = vector.from_elements %arg0 : vector<1xf32>
  return %0: vector<1xf32>
}

// CHECK-LABEL: @from_elements_3x
//  CHECK-SAME: %[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32
//       CHECK:   %[[RETVAL:.+]] = spirv.CompositeConstruct %[[ARG0]], %[[ARG1]], %[[ARG2]] : (f32, f32, f32) -> vector<3xf32>
//       CHECK:   return %[[RETVAL]]
func.func @from_elements_3x(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> vector<3xf32> {
  %0 = vector.from_elements %arg0, %arg1, %arg2 : vector<3xf32>
  return %0: vector<3xf32>
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

func.func @insert_poison_idx(%arg0 : vector<4xf32>, %arg1: f32) -> vector<4xf32> {
  // expected-error@+1 {{index -1 out of bounds for 'vector<4xf32>'}}
  %1 = vector.insert %arg1, %arg0[-1] : f32 into vector<4xf32>
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
//  CHECK-SAME: %[[V:.*]]: vector<1xf32>, %[[S:.*]]: f32
//       CHECK:   %[[R:.+]] = builtin.unrealized_conversion_cast %[[S]]
//       CHECK:   return %[[R]]
func.func @insert_size1_vector(%arg0 : vector<1xf32>, %arg1: f32) -> vector<1xf32> {
  %1 = vector.insert %arg1, %arg0[0] : f32 into vector<1xf32>
  return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL: @insert_size1_vector_dynamic
//  CHECK-SAME: %[[V:.*]]: vector<1xf32>, %[[S:.*]]: f32
//       CHECK:   %[[R:.+]] = builtin.unrealized_conversion_cast %[[S]]
//       CHECK:   return %[[R]]
func.func @insert_size1_vector_dynamic(%arg0 : vector<1xf32>, %arg1: f32, %id : index) -> vector<1xf32> {
  %1 = vector.insert %arg1, %arg0[%id] : f32 into vector<1xf32>
  return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL: @insert_dynamic
//  CHECK-SAME: %[[VAL:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[ARG2:.*]]: index
//       CHECK: %[[ID:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : index to i32
//       CHECK:   spirv.VectorInsertDynamic %[[VAL]], %[[V]][%[[ID]]] : vector<4xf32>, i32
func.func @insert_dynamic(%val: f32, %arg0 : vector<4xf32>, %id : index) -> vector<4xf32> {
  %0 = vector.insert %val, %arg0[%id] : f32 into vector<4xf32>
  return %0: vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_dynamic_cst
//  CHECK-SAME: %[[VAL:.*]]: f32, %[[V:.*]]: vector<4xf32>
//       CHECK:   spirv.CompositeInsert %[[VAL]], %[[V]][2 : i32] : f32 into vector<4xf32>
func.func @insert_dynamic_cst(%val: f32, %arg0 : vector<4xf32>) -> vector<4xf32> {
  %idx = arith.constant 2 : index
  %0 = vector.insert %val, %arg0[%idx] : f32 into vector<4xf32>
  return %0: vector<4xf32>
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

// CHECK-LABEL: @extract_element_size5_vector
func.func @extract_element_size5_vector(%arg0 : vector<5xf32>, %id : i32) -> f32 {
  // CHECK: vector.extractelement
  %0 = vector.extractelement %arg0[%id : i32] : vector<5xf32>
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_size1_vector
//  CHECK-SAME: (%[[S:.+]]: f32
func.func @extract_element_size1_vector(%arg0 : f32, %i: index) -> f32 {
  %bcast = vector.broadcast %arg0 : f32 to vector<1xf32>
  %0 = vector.extractelement %bcast[%i : index] : vector<1xf32>
  // CHECK: return %[[S]]
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_element_0d_vector
//  CHECK-SAME: (%[[S:.+]]: f32)
func.func @extract_element_0d_vector(%arg0 : f32) -> f32 {
  %bcast = vector.broadcast %arg0 : f32 to vector<f32>
  %0 = vector.extractelement %bcast[] : vector<f32>
  // CHECK: return %[[S]]
  return %0: f32
}

// -----

// CHECK-LABEL: @extract_strided_slice
//  CHECK-SAME: %[[ARG:.+]]: vector<4xf32>
//       CHECK:   spirv.VectorShuffle [1 : i32, 2 : i32] %[[ARG]], %[[ARG]] : vector<4xf32>, vector<4xf32> -> vector<2xf32>
//       CHECK:   spirv.CompositeExtract %[[ARG]][1 : i32] : vector<4xf32>
func.func @extract_strided_slice(%arg0: vector<4xf32>) -> (vector<2xf32>, vector<1xf32>) {
  %0 = vector.extract_strided_slice %arg0 {offsets = [1], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
  %1 = vector.extract_strided_slice %arg0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  return %0, %1 : vector<2xf32>, vector<1xf32>
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

// CHECK-LABEL: @insert_element_size5_vector
func.func @insert_element_size5_vector(%val: f32, %arg0 : vector<5xf32>, %id : i32) -> vector<5xf32> {
  // CHECK: vector.insertelement
  %0 = vector.insertelement %val, %arg0[%id : i32] : vector<5xf32>
  return %0 : vector<5xf32>
}

// -----

// CHECK-LABEL: @insert_element_size1_vector
//  CHECK-SAME: (%[[S:[a-z0-9]+]]: f32
func.func @insert_element_size1_vector(%scalar: f32, %vector : vector<1xf32>, %i: index) -> vector<1xf32> {
  %0 = vector.insertelement %scalar, %vector[%i : index] : vector<1xf32>
  // CHECK: %[[V:.+]] = builtin.unrealized_conversion_cast %arg0 : f32 to vector<1xf32>
  // CHECK: return %[[V]]
  return %0: vector<1xf32>
}

// -----

// CHECK-LABEL: @insert_element_0d_vector
//  CHECK-SAME: (%[[S:[a-z0-9]+]]: f32
func.func @insert_element_0d_vector(%scalar: f32, %vector : vector<f32>) -> vector<f32> {
  %0 = vector.insertelement %scalar, %vector[] : vector<f32>
  // CHECK: %[[V:.+]] = builtin.unrealized_conversion_cast %arg0 : f32 to vector<f32>
  // CHECK: return %[[V]]
  return %0: vector<f32>
}

// -----

// CHECK-LABEL: @insert_strided_slice
//  CHECK-SAME: %[[PART:.+]]: vector<2xf32>, %[[ALL:.+]]: vector<4xf32>
//       CHECK:   spirv.VectorShuffle [0 : i32, 4 : i32, 5 : i32, 3 : i32] %[[ALL]], %[[PART]] : vector<4xf32>, vector<2xf32> -> vector<4xf32>
func.func @insert_strided_slice(%arg0: vector<2xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1], strides = [1]} : vector<2xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @insert_size1_vector
//  CHECK-SAME: %[[SUB:.*]]: vector<1xf32>, %[[FULL:.*]]: vector<3xf32>
//       CHECK:   %[[S:.+]] = builtin.unrealized_conversion_cast %[[SUB]]
//       CHECK:   spirv.CompositeInsert %[[S]], %[[FULL]][2 : i32] : f32 into vector<3xf32>
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
//   CHECK-DAG:    %[[V0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//   CHECK-DAG:    %[[V1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]]
//       CHECK:    spirv.CompositeConstruct %[[V0]], %[[V1]], %[[V1]], %[[V0]] : (f32, f32, f32, f32) -> vector<4xf32>
func.func @shuffle(%v0 : vector<1xf32>, %v1: vector<1xf32>) -> vector<4xf32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 1, 0] : vector<1xf32>, vector<1xf32>
  return %shuffle : vector<4xf32>
}

// -----

// CHECK-LABEL:  func @shuffle_index_vector
//  CHECK-SAME:  %[[ARG0:.+]]: vector<1xindex>, %[[ARG1:.+]]: vector<1xindex>
//   CHECK-DAG:    %[[V0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
//   CHECK-DAG:    %[[V1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]]
//       CHECK:    spirv.CompositeConstruct %[[V0]], %[[V1]], %[[V1]], %[[V0]] : (i32, i32, i32, i32) -> vector<4xi32>
func.func @shuffle_index_vector(%v0 : vector<1xindex>, %v1: vector<1xindex>) -> vector<4xindex> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 1, 0] : vector<1xindex>, vector<1xindex>
  return %shuffle : vector<4xindex>
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
func.func @shuffle(%v0 : vector<2x16xf32>, %v1: vector<1x16xf32>) -> vector<3x16xf32> {
  // CHECK: vector.shuffle
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 2] : vector<2x16xf32>, vector<1x16xf32>
  return %shuffle : vector<3x16xf32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<1xi32>, %[[ARG1:.+]]: vector<3xi32>
//       CHECK:    %[[V0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<1xi32> to i32
//       CHECK:    %[[S1:.+]] = spirv.CompositeExtract %[[ARG1]][1 : i32] : vector<3xi32>
//       CHECK:    %[[S2:.+]] = spirv.CompositeExtract %[[ARG1]][2 : i32] : vector<3xi32>
//       CHECK:    %[[RES:.+]] = spirv.CompositeConstruct %[[V0]], %[[S1]], %[[S2]] : (i32, i32, i32) -> vector<3xi32>
//       CHECK:    return %[[RES]]
func.func @shuffle(%v0 : vector<1xi32>, %v1: vector<3xi32>) -> vector<3xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 2, 3] : vector<1xi32>, vector<3xi32>
  return %shuffle : vector<3xi32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<1xi32>
//       CHECK:    %[[V1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : vector<1xi32> to i32
//       CHECK:    %[[S0:.+]] = spirv.CompositeExtract %[[ARG0]][0 : i32] : vector<3xi32>
//       CHECK:    %[[S1:.+]] = spirv.CompositeExtract %[[ARG0]][2 : i32] : vector<3xi32>
//       CHECK:    %[[RES:.+]] = spirv.CompositeConstruct %[[S0]], %[[S1]], %[[V1]] : (i32, i32, i32) -> vector<3xi32>
//       CHECK:    return %[[RES]]
func.func @shuffle(%v0 : vector<3xi32>, %v1: vector<1xi32>) -> vector<3xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 2, 3] : vector<3xi32>, vector<1xi32>
  return %shuffle : vector<3xi32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<1xi32>, %[[ARG1:.+]]: vector<1xi32>
//   CHECK-DAG:    %[[V0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<1xi32> to i32
//   CHECK-DAG:    %[[V1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : vector<1xi32> to i32
//       CHECK:    %[[RES:.+]] = spirv.CompositeConstruct %[[V0]], %[[V1]] : (i32, i32) -> vector<2xi32>
//       CHECK:    return %[[RES]]
func.func @shuffle(%v0 : vector<1xi32>, %v1: vector<1xi32>) -> vector<2xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1] : vector<1xi32>, vector<1xi32>
  return %shuffle : vector<2xi32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>
//       CHECK:    %[[EXTR:.+]] = spirv.CompositeExtract %[[ARG0]][0 : i32] : vector<4xi32>
//       CHECK:    %[[RES:.+]]  = builtin.unrealized_conversion_cast %[[EXTR]] : i32 to vector<1xi32>
//       CHECK:    return %[[RES]] : vector<1xi32>
func.func @shuffle(%v0 : vector<4xi32>, %v1: vector<4xi32>) -> vector<1xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0] : vector<4xi32>, vector<4xi32>
  return %shuffle : vector<1xi32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>
//       CHECK:    %[[EXTR:.+]] = spirv.CompositeExtract %[[ARG1]][1 : i32] : vector<4xi32>
//       CHECK:    %[[RES:.+]]  = builtin.unrealized_conversion_cast %[[EXTR]] : i32 to vector<1xi32>
//       CHECK:    return %[[RES]] : vector<1xi32>
func.func @shuffle(%v0 : vector<4xi32>, %v1: vector<4xi32>) -> vector<1xi32> {
  %shuffle = vector.shuffle %v0, %v1 [5] : vector<4xi32>, vector<4xi32>
  return %shuffle : vector<1xi32>
}

// -----

// CHECK-LABEL:  func @shuffle
//  CHECK-SAME:  %[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>
//       CHECK:    %[[SHUFFLE:.*]] = spirv.VectorShuffle [1 : i32, -1 : i32, 5 : i32, -1 : i32] %[[ARG0]], %[[ARG1]] : vector<4xi32>, vector<4xi32> -> vector<4xi32>
//       CHECK:    return %[[SHUFFLE]] : vector<4xi32>
func.func @shuffle(%v0 : vector<4xi32>, %v1: vector<4xi32>) -> vector<4xi32> {
  %shuffle = vector.shuffle %v0, %v1 [1, -1, 5, -1] : vector<4xi32>, vector<4xi32>
  return %shuffle : vector<4xi32>
}

// -----

// CHECK-LABEL: func @interleave
//  CHECK-SAME: (%[[ARG0:.+]]: vector<2xf32>, %[[ARG1:.+]]: vector<2xf32>)
//       CHECK: %[[SHUFFLE:.*]] = spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32] %[[ARG0]], %[[ARG1]] : vector<2xf32>, vector<2xf32> -> vector<4xf32>
//       CHECK: return %[[SHUFFLE]]
func.func @interleave(%a: vector<2xf32>, %b: vector<2xf32>) -> vector<4xf32> {
  %0 = vector.interleave %a, %b : vector<2xf32> -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @interleave_size1
// CHECK-SAME: (%[[ARG0:.+]]: vector<1xf32>, %[[ARG1:.+]]: vector<1xf32>)
//  CHECK-DAG: %[[V0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<1xf32> to f32
//  CHECK-DAG: %[[V1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : vector<1xf32> to f32
//       CHECK: %[[RES:.*]] = spirv.CompositeConstruct %[[V0]], %[[V1]] : (f32, f32) -> vector<2xf32>
//       CHECK: return %[[RES]]
func.func @interleave_size1(%a: vector<1xf32>, %b: vector<1xf32>) -> vector<2xf32> {
  %0 = vector.interleave %a, %b : vector<1xf32> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @deinterleave
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xf32>)
//       CHECK: %[[SHUFFLE0:.*]] = spirv.VectorShuffle [0 : i32, 2 : i32] %[[ARG0]], %[[ARG0]] : vector<4xf32>, vector<4xf32> -> vector<2xf32>
//       CHECK: %[[SHUFFLE1:.*]] = spirv.VectorShuffle [1 : i32, 3 : i32] %[[ARG0]], %[[ARG0]] : vector<4xf32>, vector<4xf32> -> vector<2xf32>
//       CHECK: return %[[SHUFFLE0]], %[[SHUFFLE1]]
func.func @deinterleave(%a: vector<4xf32>) -> (vector<2xf32>, vector<2xf32>) {
  %0, %1 = vector.deinterleave %a : vector<4xf32> -> vector<2xf32>
  return %0, %1 : vector<2xf32>, vector<2xf32>
}

// -----

// CHECK-LABEL: func @deinterleave_scalar
// CHECK-SAME: (%[[ARG0:.+]]: vector<2xf32>)
//   CHECK-DAG: %[[EXTRACT0:.*]] = spirv.CompositeExtract %[[ARG0]][0 : i32] : vector<2xf32>
//   CHECK-DAG: %[[EXTRACT1:.*]] = spirv.CompositeExtract %[[ARG0]][1 : i32] : vector<2xf32>
//   CHECK-DAG: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[EXTRACT0]] : f32 to vector<1xf32>
//   CHECK-DAG: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[EXTRACT1]] : f32 to vector<1xf32>
//       CHECK: return %[[CAST0]], %[[CAST1]]
func.func @deinterleave_scalar(%a: vector<2xf32>) -> (vector<1xf32>, vector<1xf32>) {
  %0, %1 = vector.deinterleave %a: vector<2xf32> -> vector<1xf32>
  return %0, %1 : vector<1xf32>, vector<1xf32>
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
//       CHECK:   return %[[ADD2]]
func.func @reduction_add(%v : vector<4xi32>) -> i32 {
  %reduce = vector.reduction <add>, %v : vector<4xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: func @reduction_addf_mulf
//  CHECK-SAME:  (%[[ARG0:.+]]: vector<4xf32>, %[[ARG1:.+]]: vector<4xf32>)
//  CHECK:       %[[DOT:.+]] = spirv.Dot %[[ARG0]], %[[ARG1]] : vector<4xf32> -> f32
//  CHECK:       return %[[DOT]] : f32
func.func @reduction_addf_mulf(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> f32 {
  %mul = arith.mulf %arg0, %arg1 : vector<4xf32>
  %red = vector.reduction <add>, %mul : vector<4xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_addf_acc_mulf
//  CHECK-SAME:  (%[[ARG0:.+]]: vector<4xf32>, %[[ARG1:.+]]: vector<4xf32>, %[[ACC:.+]]: f32)
//  CHECK:       %[[DOT:.+]] = spirv.Dot %[[ARG0]], %[[ARG1]] : vector<4xf32> -> f32
//  CHECK:       %[[RES:.+]] = spirv.FAdd %[[ACC]], %[[DOT]] : f32
//  CHECK:       return %[[RES]] : f32
func.func @reduction_addf_acc_mulf(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %acc: f32) -> f32 {
  %mul = arith.mulf %arg0, %arg1 : vector<4xf32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_addf
//  CHECK-SAME:  (%[[ARG0:.+]]: vector<4xf32>)
//  CHECK:       %[[ONE:.+]] = spirv.Constant dense<1.0{{.+}}> : vector<4xf32>
//  CHECK:       %[[DOT:.+]] = spirv.Dot %[[ARG0]], %[[ONE]] : vector<4xf32> -> f32
//  CHECK:       return %[[DOT]] : f32
func.func @reduction_addf_mulf(%arg0: vector<4xf32>) -> f32 {
  %red = vector.reduction <add>, %arg0 : vector<4xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_addf_acc
//  CHECK-SAME:  (%[[ARG0:.+]]: vector<4xf32>, %[[ACC:.+]]: f32)
//  CHECK:       %[[ONE:.+]] = spirv.Constant dense<1.0{{.*}}> : vector<4xf32>
//  CHECK:       %[[DOT:.+]] = spirv.Dot %[[ARG0]], %[[ONE]] : vector<4xf32> -> f32
//  CHECK:       %[[RES:.+]] = spirv.FAdd %[[ACC]], %[[DOT]] : f32
//  CHECK:       return %[[RES]] : f32
func.func @reduction_addf_acc(%arg0: vector<4xf32>, %acc: f32) -> f32 {
  %red = vector.reduction <add>, %arg0, %acc : vector<4xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_addf_one_elem
//  CHECK-SAME:  (%[[ARG0:.+]]: vector<1xf32>)
//  CHECK:       %[[RES:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<1xf32> to f32
//  CHECK:       return %[[RES]] : f32
func.func @reduction_addf_one_elem(%arg0: vector<1xf32>) -> f32 {
  %red = vector.reduction <add>, %arg0 : vector<1xf32> into f32
  return %red : f32
}

// -----

// CHECK-LABEL: func @reduction_addf_one_elem_acc
//  CHECK-SAME:  (%[[ARG0:.+]]: vector<1xf32>, %[[ACC:.+]]: f32)
//  CHECK:       %[[RHS:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<1xf32> to f32
//  CHECK:       %[[RES:.+]] = spirv.FAdd %[[ACC]], %[[RHS]] : f32
//  CHECK:       return %[[RES]] : f32
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
//       CHECK:   return %[[MUL2]]
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
//       CHECK:   return %[[MAX2]]
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
//       CHECK:   return %[[MIN2]]
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
//       CHECK:   return %[[MAX2]]
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
//       CHECK:   return %[[MIN2]]
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
//       CHECK:   return %[[MAX2]]
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
//       CHECK:   return %[[MIN2]]
func.func @reduction_minui(%v : vector<3xi32>, %s: i32) -> i32 {
  %reduce = vector.reduction <minui>, %v, %s : vector<3xi32> into i32
  return %reduce : i32
}

// -----

// CHECK-LABEL: @shape_cast_same_type
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2xf32>)
//       CHECK:   return %[[ARG0]]
func.func @shape_cast_same_type(%arg0 : vector<2xf32>) -> vector<2xf32> {
  %1 = vector.shape_cast %arg0 : vector<2xf32> to vector<2xf32>
  return %arg0 : vector<2xf32>
}

// -----

// CHECK-LABEL: @shape_cast_size1_vector
//  CHECK-SAME: (%[[ARG0:.*]]: vector<f32>)
//       CHECK:   %[[R0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<f32> to f32
//       CHECK:   %[[R1:.+]] = builtin.unrealized_conversion_cast %[[R0]] : f32 to vector<1xf32>
//       CHECK:   return %[[R1]]
func.func @shape_cast_size1_vector(%arg0 : vector<f32>) -> vector<1xf32> {
  %1 = vector.shape_cast %arg0 : vector<f32> to vector<1xf32>
  return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL: @step()
//       CHECK:   %[[CST0:.*]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST1:.*]] = spirv.Constant 1 : i32
//       CHECK:   %[[CST2:.*]] = spirv.Constant 2 : i32
//       CHECK:   %[[CST3:.*]] = spirv.Constant 3 : i32
//       CHECK:   %[[CONSTRUCT:.*]] = spirv.CompositeConstruct %[[CST0]], %[[CST1]], %[[CST2]], %[[CST3]] : (i32, i32, i32, i32) -> vector<4xi32>
//       CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[CONSTRUCT]] : vector<4xi32> to vector<4xindex>
//       CHECK:   return %[[CAST]] : vector<4xindex>
func.func @step() -> vector<4xindex> {
  %0 = vector.step : vector<4xindex>
  return %0 : vector<4xindex>
}

// -----

// CHECK-LABEL: @step_size1()
//       CHECK:   %[[CST0:.*]] = spirv.Constant 0 : i32
//       CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[CST0]] : i32 to vector<1xindex>
//       CHECK:   return %[[CAST]] : vector<1xindex>
func.func @step_size1() -> vector<1xindex> {
  %0 = vector.step : vector<1xindex>
  return %0 : vector<1xindex>
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  } {

// CHECK-LABEL: @vector_load
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4xf32, #spirv.storage_class<StorageBuffer>>)
//       CHECK:   %[[S0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[S1:.+]] = builtin.unrealized_conversion_cast %[[C0]] : index to i32
//       CHECK:   %[[CST1:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST2:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST3:.+]] = spirv.Constant 1 : i32
//       CHECK:   %[[S4:.+]] = spirv.AccessChain %[[S0]][%[[CST1]], %[[S1]]] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK:   %[[S5:.+]] = spirv.Bitcast %[[S4]] : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<vector<4xf32>, StorageBuffer>
//       CHECK:   %[[R0:.+]] = spirv.Load "StorageBuffer" %[[S5]] : vector<4xf32>
//       CHECK:   return %[[R0]] : vector<4xf32>
func.func @vector_load(%arg0 : memref<4xf32, #spirv.storage_class<StorageBuffer>>) -> vector<4xf32> {
  %idx = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = vector.load %arg0[%idx] : memref<4xf32, #spirv.storage_class<StorageBuffer>>, vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL: @vector_load_2d
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4x4xf32, #spirv.storage_class<StorageBuffer>>) -> vector<4xf32> {
//       CHECK:   %[[S0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[S1:.+]] = builtin.unrealized_conversion_cast %[[C0]] : index to i32
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[S2:.+]] = builtin.unrealized_conversion_cast %[[C1]] : index to i32
//       CHECK:   %[[CST0_1:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST0_2:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST4:.+]] = spirv.Constant 4 : i32
//       CHECK:   %[[S3:.+]] = spirv.IMul %[[S1]], %[[CST4]] : i32
//       CHECK:   %[[CST1:.+]] = spirv.Constant 1 : i32
//       CHECK:   %[[S6:.+]] = spirv.IAdd  %[[S2]], %[[S3]] : i32
//       CHECK:   %[[S7:.+]] = spirv.AccessChain %[[S0]][%[[CST0_1]], %[[S6]]] : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK:   %[[S8:.+]] = spirv.Bitcast %[[S7]] : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<vector<4xf32>, StorageBuffer>
//       CHECK:   %[[R0:.+]] = spirv.Load "StorageBuffer" %[[S8]] : vector<4xf32>
//       CHECK:   return %[[R0]] : vector<4xf32>
func.func @vector_load_2d(%arg0 : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>) -> vector<4xf32> {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %0 = vector.load %arg0[%idx_0, %idx_1] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>, vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL: @vector_store
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4xf32, #spirv.storage_class<StorageBuffer>>
//  CHECK-SAME:  %[[ARG1:.*]]: vector<4xf32>
//       CHECK:   %[[S0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[S1:.+]] = builtin.unrealized_conversion_cast %[[C0]] : index to i32
//       CHECK:   %[[CST1:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST2:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST3:.+]] = spirv.Constant 1 : i32
//       CHECK:   %[[S4:.+]] = spirv.AccessChain %[[S0]][%[[CST1]], %[[S1]]] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK:   %[[S5:.+]] = spirv.Bitcast %[[S4]] : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<vector<4xf32>, StorageBuffer>
//       CHECK:   spirv.Store "StorageBuffer" %[[S5]], %[[ARG1]] : vector<4xf32>
func.func @vector_store(%arg0 : memref<4xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : vector<4xf32>) {
  %idx = arith.constant 0 : index
  vector.store %arg1, %arg0[%idx] : memref<4xf32, #spirv.storage_class<StorageBuffer>>, vector<4xf32>
  return
}

// CHECK-LABEL: @vector_store_2d
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
//  CHECK-SAME:  %[[ARG1:.*]]: vector<4xf32>
//       CHECK:   %[[S0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[S1:.+]] = builtin.unrealized_conversion_cast %[[C0]] : index to i32
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[S2:.+]] = builtin.unrealized_conversion_cast %[[C1]] : index to i32
//       CHECK:   %[[CST0_1:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST0_2:.+]] = spirv.Constant 0 : i32
//       CHECK:   %[[CST4:.+]] = spirv.Constant 4 : i32
//       CHECK:   %[[S3:.+]] = spirv.IMul %[[S1]], %[[CST4]] : i32
//       CHECK:   %[[CST1:.+]] = spirv.Constant 1 : i32
//       CHECK:   %[[S6:.+]] = spirv.IAdd %[[S2]], %[[S3]] : i32
//       CHECK:   %[[S7:.+]] = spirv.AccessChain %[[S0]][%[[CST0_1]], %[[S6]]] : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK:   %[[S8:.+]] = spirv.Bitcast %[[S7]] : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<vector<4xf32>, StorageBuffer>
//       CHECK:   spirv.Store "StorageBuffer" %[[S8]], %[[ARG1]] : vector<4xf32>
func.func @vector_store_2d(%arg0 : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : vector<4xf32>) {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  vector.store %arg1, %arg0[%idx_0, %idx_1] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>, vector<4xf32>
  return
}

} // end module
