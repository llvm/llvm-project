// RUN: mlir-opt -split-input-file -convert-vector-to-spirv -verify-diagnostics %s -o - | FileCheck %s

// CHECK-LABEL: broadcast
//  CHECK-SAME: %[[A:.*]]: f32
//       CHECK:   spv.CompositeConstruct %[[A]], %[[A]], %[[A]], %[[A]] : vector<4xf32>
//       CHECK:   spv.CompositeConstruct %[[A]], %[[A]] : vector<2xf32>
func @broadcast(%arg0 : f32) {
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %1 = vector.broadcast %arg0 : f32 to vector<2xf32>
  spv.Return
}

// -----

// CHECK-LABEL: extract_insert
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>
//       CHECK:   %[[S:.*]] = spv.CompositeExtract %[[V]][1 : i32] : vector<4xf32>
//       CHECK:   spv.CompositeInsert %[[S]], %[[V]][0 : i32] : f32 into vector<4xf32>
func @extract_insert(%arg0 : vector<4xf32>) {
  %0 = vector.extract %arg0[1] : vector<4xf32>
  %1 = vector.insert %0, %arg0[0] : f32 into vector<4xf32>
  spv.Return
}

// -----

// CHECK-LABEL: extract_element
//  CHECK-SAME: %[[V:.*]]: vector<4xf32>, %[[ID:.*]]: i32
//       CHECK:   spv.VectorExtractDynamic %[[V]][%[[ID]]] : vector<4xf32>, i32
func @extract_element(%arg0 : vector<4xf32>, %id : i32) {
  %0 = vector.extractelement %arg0[%id : i32] : vector<4xf32>
  spv.ReturnValue %0: f32
}

// -----

func @extract_element_negative(%arg0 : vector<5xf32>, %id : i32) {
// expected-error @+1 {{failed to legalize operation 'vector.extractelement'}}
  %0 = vector.extractelement %arg0[%id : i32] : vector<5xf32>
  spv.ReturnValue %0: f32
}

// -----

// CHECK-LABEL: insert_element
//  CHECK-SAME: %[[VAL:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[ID:.*]]: i32
//       CHECK:   spv.VectorInsertDynamic %[[VAL]], %[[V]][%[[ID]]] : vector<4xf32>, i32
func @insert_element(%val: f32, %arg0 : vector<4xf32>, %id : i32) {
  %0 = vector.insertelement %val, %arg0[%id : i32] : vector<4xf32>
  spv.ReturnValue %0: vector<4xf32>
}

// -----

func @insert_element_negative(%val: f32, %arg0 : vector<5xf32>, %id : i32) {
// expected-error @+1 {{failed to legalize operation 'vector.insertelement'}}
  %0 = vector.insertelement %val, %arg0[%id : i32] : vector<5xf32>
  spv.Return
}
