// RUN: mlir-opt  --convert-vector-to-llvm="enable-one-to-n-conversion=true" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @vector_extract_vector(
// CHECK-SAME: %[[ARG0:.+]]: vector<4x4xf32>
func.func @vector_extract_vector(%arg0: vector<4x4xf32>) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
    // CHECK-NEXT: %[[CAST:.+]]:4 = builtin.unrealized_conversion_cast %[[ARG0]] : vector<4x4xf32> to vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
    %0 = vector.extract %arg0[0] : vector<4xf32> from vector<4x4xf32>
    %1 = vector.extract %arg0[1] : vector<4xf32> from vector<4x4xf32>
    %2 = vector.extract %arg0[2] : vector<4xf32> from vector<4x4xf32>
    %3 = vector.extract %arg0[3] : vector<4xf32> from vector<4x4xf32>
    // CHECK-NEXT: return %[[CAST]]#3, %[[CAST]]#2, %[[CAST]]#1, %[[CAST]]#0
    return %3, %2, %1, %0 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}

// -----

// CHECK-LABEL: func @vector_extract_linearize(
// CHECK-SAME: %[[ARG0:.+]]: vector<5x4x3xf32>
func.func @vector_extract_linearize(%arg0: vector<5x4x3xf32>) -> (vector<3xf32>, vector<3xf32>, vector<3xf32>) {
    // CHECK-NEXT: %[[CAST:.+]]:20 = builtin.unrealized_conversion_cast %[[ARG0]] : vector<5x4x3xf32> to vector<3xf32>
    %0 = vector.extract %arg0[0, 0] : vector<3xf32> from vector<5x4x3xf32>
    %1 = vector.extract %arg0[0, 1] : vector<3xf32> from vector<5x4x3xf32>
    %2 = vector.extract %arg0[1, 1] : vector<3xf32> from vector<5x4x3xf32>
    // CHECK-NEXT: return %[[CAST]]#0, %[[CAST]]#1, %[[CAST]]#5
    return %0, %1, %2 : vector<3xf32>, vector<3xf32>, vector<3xf32>
}

// -----

// CHECK-LABEL: func @vector_extract_scalar(
// CHECK-SAME: %[[ARG0:.+]]: vector<2x2xf32>
func.func @vector_extract_scalar(%arg0: vector<2x2xf32>) -> (f32) {
  // CHECK: %[[CAST:.+]]:2 = builtin.unrealized_conversion_cast %[[ARG0]] : vector<2x2xf32> to vector<2xf32>, vector<2xf32>
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[EXTRACTED:.+]] = llvm.extractelement %[[CAST]]#0[%[[C0]] : i64]
  %0 = vector.extract %arg0[0, 0] : f32 from vector<2x2xf32>
  // CHECK: return %[[EXTRACTED]]
  return %0 : f32
}

// -----

// CHECK-LABEL: func @vector_extract_lhs_multiple(
// CHECK-SAME: %[[ARG0:.+]]: vector<2x2x2xf32>)
func.func @vector_extract_lhs_multiple(%arg0: vector<2x2x2xf32>) -> vector<2x2xf32> {
  // CHECK: %[[CAST:.+]]:4 = builtin.unrealized_conversion_cast %[[ARG0]] : vector<2x2x2xf32> to vector<2xf32>
  // CHECK: %[[SELECTED:.+]] = builtin.unrealized_conversion_cast %[[CAST]]#0, %[[CAST]]#1 : vector<2xf32>, vector<2xf32>
  %0 = vector.extract %arg0[0] : vector<2x2xf32> from vector<2x2x2xf32>
  // CHECK: return %[[SELECTED]]
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: func @vector_extract_rank_0(
// CHECK-SAME: %[[ARG0:.+]]: vector<f32>)
func.func @vector_extract_rank_0(%arg0: vector<f32>) -> f32 {
  // CHECK: %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<f32> to vector<1xf32>
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[ELEM:.+]] = llvm.extractelement %[[CAST]][%[[C0]] : i64]
  %0 = vector.extract %arg0[] : f32 from vector<f32>
  // CHECK: return %[[ELEM]]
  return %0 : f32
}

// -----

// CHECK-LABEL: func @vector_insert_vector(
// CHECK-SAME: %[[VAL0:.+]]: vector<4xf32>, %[[VAL1:.+]]: vector<4xf32>, %[[VAL2:.+]]: vector<4xf32>, %[[AGG:.+]]: vector<4x4xf32>)
func.func @vector_insert_vector(%val0: vector<4xf32>, %val1: vector<4xf32>, %val2: vector<4xf32>, %agg: vector<4x4xf32>) -> (vector<4x4xf32>) {
  // CHECK: %[[CAST:.+]]:4 = builtin.unrealized_conversion_cast %[[AGG]] : vector<4x4xf32> to vector<4xf32>
  %0 = vector.insert %val0, %agg[0] : vector<4xf32> into vector<4x4xf32>
  %1 = vector.insert %val1, %0[1] : vector<4xf32> into vector<4x4xf32>
  %2 = vector.insert %val2, %1[2] : vector<4xf32> into vector<4x4xf32>

  // CHECK: %[[INSERTION_CAST:.+]] = builtin.unrealized_conversion_cast %[[VAL0]], %[[VAL1]], %[[VAL2]], %[[CAST]]#3
  // CHECK: return %[[INSERTION_CAST]]
  return %2 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func @vector_insert_linearize(
// CHECK-SAME: %[[VAL:.+]]: vector<3xf32>, %[[AGG:.+]]: vector<5x4x3xf32>)
func.func @vector_insert_linearize(%val: vector<3xf32>, %agg: vector<5x4x3xf32>) -> (vector<5x4x3xf32>) {
  // CHECK: %[[CAST:.+]]:20 = builtin.unrealized_conversion_cast %[[AGG]] : vector<5x4x3xf32> to vector<3xf32>

  %0 = vector.insert %val, %agg[0, 0] : vector<3xf32> into vector<5x4x3xf32>
  %1 = vector.insert %val, %0[0, 1] : vector<3xf32> into vector<5x4x3xf32>
  %2 = vector.insert %val, %1[1, 1] : vector<3xf32> into vector<5x4x3xf32>

  // CHECK: %[[INSERTION_CAST:.+]] = builtin.unrealized_conversion_cast %[[VAL]], %[[VAL]], %[[CAST]]#2, %[[CAST]]#3, %[[CAST]]#4, %[[VAL]]
  // CHECK: return %[[INSERTION_CAST]]
  return %2 : vector<5x4x3xf32>
}

// -----

// CHECK-LABEL: func @vector_insert_scalar(
// CHECK-SAME: %[[VAL:.+]]: f32, %[[AGG:.+]]: vector<2x2xf32>)
func.func @vector_insert_scalar(%val: f32, %agg: vector<2x2xf32>) -> (vector<2x2xf32>) {
  // CHECK-DAG: %[[CAST:.+]]:2 = builtin.unrealized_conversion_cast %[[AGG]] : vector<2x2xf32> to vector<2xf32>
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[MODIFIED_VECTOR:.+]] = llvm.insertelement %[[VAL]], %[[CAST]]#0[%[[C1]] : i64]
  // CHECK: %[[INSERTION_CAST:.+]] = builtin.unrealized_conversion_cast %[[MODIFIED_VECTOR]], %[[CAST]]#1
  %0 = vector.insert %val, %agg[0, 1] : f32 into vector<2x2xf32>

  // CHECK: return %[[INSERTION_CAST]]
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: func @vector_insert_sources_multiple(
// CHECK-SAME: %[[TO_STORE:.+]]: vector<2x2xf32>,
// CHECK-SAME: %[[DEST:.+]]: vector<2x2x2xf32>
func.func @vector_insert_sources_multiple(%val: vector<2x2xf32>, %dest: vector<2x2x2xf32>) -> (vector<2x2x2xf32>) {
  // CHECK: %[[CAST_DEST:.+]]:4 = builtin.unrealized_conversion_cast %[[DEST]] : vector<2x2x2xf32> to vector<2xf32>
  // CHECK: %[[CAST_TO_STORE:.+]]:2 = builtin.unrealized_conversion_cast %[[TO_STORE]] : vector<2x2xf32> to vector<2xf32>
  // CHECK: %[[INSERT:.+]] = builtin.unrealized_conversion_cast %[[CAST_TO_STORE]]#0, %[[CAST_TO_STORE]]#1, %[[CAST_DEST]]#2, %[[CAST_DEST]]#3

  %0 = vector.insert %val, %dest[0] : vector<2x2xf32> into vector<2x2x2xf32>
  // CHECK: return %[[INSERT]]
  return %0 : vector<2x2x2xf32>
}

// -----

// CHECK-LABEL: func @vector_insert_rank_0(
// CHECK-SAME: %[[TO_STORE:.+]]: f32,
// CHECK-SAME: %[[DEST:.+]]: vector<f32>
func.func @vector_insert_rank_0(%val: f32, %dest: vector<f32>) -> (vector<f32>) {
  // CHECK: %[[CAST_DEST:.+]] = builtin.unrealized_conversion_cast %[[DEST]] : vector<f32> to vector<1xf32>
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[INSERT:.+]] = llvm.insertelement %[[TO_STORE]], %[[CAST_DEST]][%[[C0]] : i64]
  // CHECK: %[[INSERT_CAST:.+]] = builtin.unrealized_conversion_cast %[[INSERT]] : vector<1xf32> to vector<f32>
  %0 = vector.insert %val, %dest[] : f32 into vector<f32>
  // CHECK: return %[[INSERT_CAST]]
  return %0 : vector<f32>
}
