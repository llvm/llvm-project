// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.CompositeConstruct
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @composite_construct_vector
func.func @composite_construct_vector(%arg0: f32, %arg1: f32, %arg2 : f32) -> vector<3xf32> {
  // CHECK: spirv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : (f32, f32, f32) -> vector<3xf32>
  %0 = spirv.CompositeConstruct %arg0, %arg1, %arg2 : (f32, f32, f32) -> vector<3xf32>
  return %0: vector<3xf32>
}

// CHECK-LABEL: func @composite_construct_struct
func.func @composite_construct_struct(%arg0: vector<3xf32>, %arg1: !spirv.array<4xf32>, %arg2 : !spirv.struct<(f32)>) -> !spirv.struct<(vector<3xf32>, !spirv.array<4xf32>, !spirv.struct<(f32)>)> {
  // CHECK: spirv.CompositeConstruct
  %0 = spirv.CompositeConstruct %arg0, %arg1, %arg2 : (vector<3xf32>, !spirv.array<4xf32>, !spirv.struct<(f32)>) -> !spirv.struct<(vector<3xf32>, !spirv.array<4xf32>, !spirv.struct<(f32)>)>
  return %0: !spirv.struct<(vector<3xf32>, !spirv.array<4xf32>, !spirv.struct<(f32)>)>
}

// CHECK-LABEL: func @composite_construct_mixed_scalar_vector
func.func @composite_construct_mixed_scalar_vector(%arg0: f32, %arg1: f32, %arg2 : vector<2xf32>) -> vector<4xf32> {
  // CHECK: spirv.CompositeConstruct %{{.+}}, %{{.+}}, %{{.+}} : (f32, vector<2xf32>, f32) -> vector<4xf32>
  %0 = spirv.CompositeConstruct %arg0, %arg2, %arg1 : (f32, vector<2xf32>, f32) -> vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL: func @composite_construct_coopmatrix_khr
func.func @composite_construct_coopmatrix_khr(%arg0 : f32) -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA> {
  // CHECK: spirv.CompositeConstruct {{%.*}} : (f32) -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA>
  %0 = spirv.CompositeConstruct %arg0 : (f32) -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA>
  return %0: !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA>
}

// CHECK-LABEL: func @composite_construct_coopmatrix_nv
func.func @composite_construct_coopmatrix_nv(%arg0 : f32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup> {
  // CHECK: spirv.CompositeConstruct {{%.*}} : (f32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup>
  %0 = spirv.CompositeConstruct %arg0 : (f32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup>
  return %0: !spirv.NV.coopmatrix<8x16xf32, Subgroup>
}

// -----

func.func @composite_construct_invalid_result_type(%arg0: f32, %arg1: f32, %arg2 : f32) -> vector<3xf32> {
  // expected-error @+1 {{has incorrect number of operands: expected 3, but provided 2}}
  %0 = spirv.CompositeConstruct %arg0, %arg2 : (f32, f32) -> vector<3xf32>
  return %0: vector<3xf32>
}

// -----

func.func @composite_construct_invalid_operand_type(%arg0: f32, %arg1: f32, %arg2 : f32) -> vector<3xi32> {
  // expected-error @+1 {{operand type mismatch: expected operand type 'i32', but provided 'f32'}}
  %0 = spirv.CompositeConstruct %arg0, %arg1, %arg2 : (f32, f32, f32) -> vector<3xi32>
  return %0: vector<3xi32>
}

// -----

func.func @composite_construct_khr_coopmatrix_incorrect_operand_count(%arg0 : f32, %arg1 : f32) ->
  !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA> {
  // expected-error @+1 {{has incorrect number of operands: expected 1, but provided 2}}
  %0 = spirv.CompositeConstruct %arg0, %arg1 : (f32, f32) -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA>
  return %0: !spirv.coopmatrix<8x16xf32, Subgroup, MatrixA>
}

// -----

func.func @composite_construct_khr_coopmatrix_incorrect_element_type(%arg0 : i32) ->
  !spirv.coopmatrix<8x16xf32, Subgroup, MatrixB> {
  // expected-error @+1 {{operand type mismatch: expected operand type 'f32', but provided 'i32'}}
  %0 = spirv.CompositeConstruct %arg0 : (i32) -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixB>
  return %0: !spirv.coopmatrix<8x16xf32, Subgroup, MatrixB>
}

// -----

func.func @composite_construct_NV.coopmatrix_incorrect_operand_count(%arg0 : f32, %arg1 : f32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup> {
  // expected-error @+1 {{has incorrect number of operands: expected 1, but provided 2}}
  %0 = spirv.CompositeConstruct %arg0, %arg1 : (f32, f32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup>
  return %0: !spirv.NV.coopmatrix<8x16xf32, Subgroup>
}

// -----

func.func @composite_construct_NV.coopmatrix_incorrect_element_type(%arg0 : i32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup> {
  // expected-error @+1 {{operand type mismatch: expected operand type 'f32', but provided 'i32'}}
  %0 = spirv.CompositeConstruct %arg0 : (i32) -> !spirv.NV.coopmatrix<8x16xf32, Subgroup>
  return %0: !spirv.NV.coopmatrix<8x16xf32, Subgroup>
}

// -----

func.func @composite_construct_array(%arg0: f32) -> !spirv.array<4xf32> {
  // expected-error @+1 {{expected to return a vector or cooperative matrix when the number of constituents is less than what the result needs}}
  %0 = spirv.CompositeConstruct %arg0 : (f32) -> !spirv.array<4xf32>
  return %0: !spirv.array<4xf32>
}

// -----

func.func @composite_construct_vector_wrong_element_type(%arg0: f32, %arg1: f32, %arg2 : vector<2xi32>) -> vector<4xf32> {
  // expected-error @+1 {{operand element type mismatch: expected to be 'f32', but provided 'i32'}}
  %0 = spirv.CompositeConstruct %arg0, %arg2, %arg1 : (f32, vector<2xi32>, f32) -> vector<4xf32>
  return %0: vector<4xf32>
}

// -----

func.func @composite_construct_vector_wrong_count(%arg0: f32, %arg1: f32, %arg2 : vector<2xf32>) -> vector<4xf32> {
  // expected-error @+1 {{op has incorrect number of operands: expected 4, but provided 3}}
  %0 = spirv.CompositeConstruct %arg0, %arg2 : (f32, vector<2xf32>) -> vector<4xf32>
  return %0: vector<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CompositeExtractOp
//===----------------------------------------------------------------------===//

func.func @composite_extract_array(%arg0: !spirv.array<4xf32>) -> f32 {
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32] : !spirv.array<4 x f32>
  %0 = spirv.CompositeExtract %arg0[1 : i32] : !spirv.array<4xf32>
  return %0: f32
}

// -----

func.func @composite_extract_struct(%arg0 : !spirv.struct<(f32, !spirv.array<4xf32>)>) -> f32 {
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32, 2 : i32] : !spirv.struct<(f32, !spirv.array<4 x f32>)>
  %0 = spirv.CompositeExtract %arg0[1 : i32, 2 : i32] : !spirv.struct<(f32, !spirv.array<4xf32>)>
  return %0 : f32
}

// -----

func.func @composite_extract_vector(%arg0 : vector<4xf32>) -> f32 {
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32] : vector<4xf32>
  %0 = spirv.CompositeExtract %arg0[1 : i32] : vector<4xf32>
  return %0 : f32
}

// -----

func.func @composite_extract_NV.coopmatrix(%arg0 : !spirv.NV.coopmatrix<8x16xf32, Subgroup>) -> f32 {
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[2 : i32] : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
  %0 = spirv.CompositeExtract %arg0[2 : i32] : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
  return %0 : f32
}

// -----

func.func @composite_extract_no_ssa_operand() -> () {
  // expected-error @+1 {{expected SSA operand}}
  %0 = spirv.CompositeExtract [4 : i32, 1 : i32] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_invalid_index_type_1() -> () {
  %0 = spirv.Constant 10 : i32
  %1 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  %2 = spirv.Load "Function" %1 ["Volatile"] : !spirv.array<4x!spirv.array<4xf32>>
  // expected-error @+1 {{expected attribute value}}
  %3 = spirv.CompositeExtract %2[%0] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_invalid_index_type_2(%arg0 : !spirv.array<4x!spirv.array<4xf32>>) -> () {
  // expected-error @+1 {{attribute 'indices' failed to satisfy constraint: 32-bit integer array attribute}}
  %0 = spirv.CompositeExtract %arg0[1] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_invalid_index_identifier(%arg0 : !spirv.array<4x!spirv.array<4xf32>>) -> () {
  // expected-error @+1 {{expected attribute value}}
  %0 = spirv.CompositeExtract %arg0 ]1 : i32) : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_2D_array_out_of_bounds_access_1(%arg0: !spirv.array<4x!spirv.array<4xf32>>) -> () {
  // expected-error @+1 {{index 4 out of bounds for '!spirv.array<4 x !spirv.array<4 x f32>>'}}
  %0 = spirv.CompositeExtract %arg0[4 : i32, 1 : i32] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_2D_array_out_of_bounds_access_2(%arg0: !spirv.array<4x!spirv.array<4xf32>>
) -> () {
  // expected-error @+1 {{index 4 out of bounds for '!spirv.array<4 x f32>'}}
  %0 = spirv.CompositeExtract %arg0[1 : i32, 4 : i32] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_struct_element_out_of_bounds_access(%arg0 : !spirv.struct<(f32, !spirv.array<4xf32>)>) -> () {
  // expected-error @+1 {{index 2 out of bounds for '!spirv.struct<(f32, !spirv.array<4 x f32>)>'}}
  %0 = spirv.CompositeExtract %arg0[2 : i32, 0 : i32] : !spirv.struct<(f32, !spirv.array<4xf32>)>
  return
}

// -----

func.func @composite_extract_vector_out_of_bounds_access(%arg0: vector<4xf32>) -> () {
  // expected-error @+1 {{index 4 out of bounds for 'vector<4xf32>'}}
  %0 = spirv.CompositeExtract %arg0[4 : i32] : vector<4xf32>
  return
}

// -----

func.func @composite_extract_invalid_types_1(%arg0: !spirv.array<4x!spirv.array<4xf32>>) -> () {
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 3}}
  %0 = spirv.CompositeExtract %arg0[1 : i32, 2 : i32, 3 : i32] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_invalid_types_2(%arg0: f32) -> () {
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 1}}
  %0 = spirv.CompositeExtract %arg0[1 : i32] : f32
  return
}

// -----

func.func @composite_extract_invalid_extracted_type(%arg0: !spirv.array<4x!spirv.array<4xf32>>) -> () {
  // expected-error @+1 {{expected at least one index for spirv.CompositeExtract}}
  %0 = spirv.CompositeExtract %arg0[] : !spirv.array<4x!spirv.array<4xf32>>
  return
}

// -----

func.func @composite_extract_result_type_mismatch(%arg0: !spirv.array<4xf32>) -> i32 {
  // expected-error @+1 {{invalid result type: expected 'f32' but provided 'i32'}}
  %0 = "spirv.CompositeExtract"(%arg0) {indices = [2: i32]} : (!spirv.array<4xf32>) -> (i32)
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CompositeInsert
//===----------------------------------------------------------------------===//

func.func @composite_insert_array(%arg0: !spirv.array<4xf32>, %arg1: f32) -> !spirv.array<4xf32> {
  // CHECK: {{%.*}} = spirv.CompositeInsert {{%.*}}, {{%.*}}[1 : i32] : f32 into !spirv.array<4 x f32>
  %0 = spirv.CompositeInsert %arg1, %arg0[1 : i32] : f32 into !spirv.array<4xf32>
  return %0: !spirv.array<4xf32>
}

// -----

func.func @composite_insert_struct(%arg0: !spirv.struct<(!spirv.array<4xf32>, f32)>, %arg1: !spirv.array<4xf32>) -> !spirv.struct<(!spirv.array<4xf32>, f32)> {
  // CHECK: {{%.*}} = spirv.CompositeInsert {{%.*}}, {{%.*}}[0 : i32] : !spirv.array<4 x f32> into !spirv.struct<(!spirv.array<4 x f32>, f32)>
  %0 = spirv.CompositeInsert %arg1, %arg0[0 : i32] : !spirv.array<4xf32> into !spirv.struct<(!spirv.array<4xf32>, f32)>
  return %0: !spirv.struct<(!spirv.array<4xf32>, f32)>
}

// -----

func.func @composite_insert_NV.coopmatrix(%arg0: !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %arg1: i32) -> !spirv.NV.coopmatrix<8x16xi32, Subgroup> {
  // CHECK: {{%.*}} = spirv.CompositeInsert {{%.*}}, {{%.*}}[5 : i32] : i32 into !spirv.NV.coopmatrix<8x16xi32, Subgroup>
  %0 = spirv.CompositeInsert %arg1, %arg0[5 : i32] : i32 into !spirv.NV.coopmatrix<8x16xi32, Subgroup>
  return %0: !spirv.NV.coopmatrix<8x16xi32, Subgroup>
}

// -----

func.func @composite_insert_no_indices(%arg0: !spirv.array<4xf32>, %arg1: f32) -> !spirv.array<4xf32> {
  // expected-error @+1 {{expected at least one index}}
  %0 = spirv.CompositeInsert %arg1, %arg0[] : f32 into !spirv.array<4xf32>
  return %0: !spirv.array<4xf32>
}

// -----

func.func @composite_insert_out_of_bounds(%arg0: !spirv.array<4xf32>, %arg1: f32) -> !spirv.array<4xf32> {
  // expected-error @+1 {{index 4 out of bounds}}
  %0 = spirv.CompositeInsert %arg1, %arg0[4 : i32] : f32 into !spirv.array<4xf32>
  return %0: !spirv.array<4xf32>
}

// -----

func.func @composite_insert_invalid_object_type(%arg0: !spirv.array<4xf32>, %arg1: f64) -> !spirv.array<4xf32> {
  // expected-error @+1 {{object operand type should be 'f32', but found 'f64'}}
  %0 = spirv.CompositeInsert %arg1, %arg0[3 : i32] : f64 into !spirv.array<4xf32>
  return %0: !spirv.array<4xf32>
}

// -----

func.func @composite_insert_invalid_result_type(%arg0: !spirv.array<4xf32>, %arg1 : f32) -> !spirv.array<4xf64> {
  // expected-error @+1 {{result type should be the same as the composite type, but found '!spirv.array<4 x f32>' vs '!spirv.array<4 x f64>'}}
  %0 = "spirv.CompositeInsert"(%arg1, %arg0) {indices = [0: i32]} : (f32, !spirv.array<4xf32>) -> !spirv.array<4xf64>
  return %0: !spirv.array<4xf64>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.VectorExtractDynamic
//===----------------------------------------------------------------------===//

func.func @vector_dynamic_extract(%vec: vector<4xf32>, %id : i32) -> f32 {
  // CHECK: spirv.VectorExtractDynamic %{{.*}}[%{{.*}}] : vector<4xf32>, i32
  %0 = spirv.VectorExtractDynamic %vec[%id] : vector<4xf32>, i32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// spirv.VectorInsertDynamic
//===----------------------------------------------------------------------===//

func.func @vector_dynamic_insert(%val: f32, %vec: vector<4xf32>, %id : i32) -> vector<4xf32> {
  // CHECK: spirv.VectorInsertDynamic %{{.*}}, %{{.*}}[%{{.*}}] : vector<4xf32>, i32
  %0 = spirv.VectorInsertDynamic %val, %vec[%id] : vector<4xf32>, i32
  return %0 : vector<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.VectorShuffle
//===----------------------------------------------------------------------===//

func.func @vector_shuffle(%vector1: vector<4xf32>, %vector2: vector<2xf32>) -> vector<3xf32> {
  // CHECK: %{{.+}} = spirv.VectorShuffle [1 : i32, 3 : i32, -1 : i32] %{{.+}}, %arg1 : vector<4xf32>, vector<2xf32> -> vector<3xf32>
  %0 = spirv.VectorShuffle [1: i32, 3: i32, 0xffffffff: i32] %vector1, %vector2 : vector<4xf32>, vector<2xf32> -> vector<3xf32>
  return %0: vector<3xf32>
}

// -----

func.func @vector_shuffle_extra_selector(%vector1: vector<4xf32>, %vector2: vector<2xf32>) -> vector<3xf32> {
  // expected-error @+1 {{result type element count (3) mismatch with the number of component selectors (4)}}
  %0 = spirv.VectorShuffle [1: i32, 3: i32, 5: i32, 2: i32] %vector1, %vector2 : vector<4xf32>, vector<2xf32> -> vector<3xf32>
  return %0: vector<3xf32>
}

// -----

func.func @vector_shuffle_extra_selector(%vector1: vector<4xf32>, %vector2: vector<2xf32>) -> vector<3xf32> {
  // expected-error @+1 {{component selector 7 out of range: expected to be in [0, 6) or 0xffffffff}}
  %0 = spirv.VectorShuffle [1: i32, 7: i32, 5: i32] %vector1, %vector2 : vector<4xf32>, vector<2xf32> -> vector<3xf32>
  return %0: vector<3xf32>
}
