// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @broadcast_to_scalar(%arg0: f32) -> f32 {
  // expected-error@+1 {{custom op 'vector.broadcast' invalid kind of type specified}}
  %0 = vector.broadcast %arg0 : f32 to f32
}

// -----

func.func @broadcast_rank_too_high(%arg0: vector<4x4xf32>) {
  // expected-error@+1 {{'vector.broadcast' op source rank higher than destination rank}}
  %1 = vector.broadcast %arg0 : vector<4x4xf32> to vector<4xf32>
}

// -----

func.func @broadcast_rank_too_high_0d(%arg0: vector<1xf32>) {
  // expected-error@+1 {{'vector.broadcast' op source rank higher than destination rank}}
  %1 = vector.broadcast %arg0 : vector<1xf32> to vector<f32>
}

// -----

func.func @broadcast_dim1_mismatch(%arg0: vector<7xf32>) {
  // expected-error@+1 {{'vector.broadcast' op dimension mismatch (7 vs. 3)}}
  %1 = vector.broadcast %arg0 : vector<7xf32> to vector<3xf32>
}

// -----

func.func @broadcast_dim2_mismatch(%arg0: vector<4x8xf32>) {
  // expected-error@+1 {{'vector.broadcast' op dimension mismatch (4 vs. 1)}}
  %1 = vector.broadcast %arg0 : vector<4x8xf32> to vector<1x8xf32>
}

// -----

func.func @broadcast_unknown(%arg0: memref<4x8xf32>) {
  // expected-error@+1 {{'vector.broadcast' op source type is not a vector}}
  %1 = vector.broadcast %arg0 : memref<4x8xf32> to vector<1x8xf32>
}

// -----

func.func @shuffle_elt_type_mismatch(%arg0: vector<2xf32>, %arg1: vector<2xi32>) {
  // expected-error@+1 {{'vector.shuffle' op failed to verify that second operand v2 and result have same element type}}
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xf32>, vector<2xi32>
}

// -----

func.func @shuffle_rank_mismatch(%arg0: vector<2xf32>, %arg1: vector<4x2xf32>) {
  // expected-error@+1 {{'vector.shuffle' op rank mismatch}}
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xf32>, vector<4x2xf32>
}

// -----

func.func @shuffle_trailing_dim_size_mismatch(%arg0: vector<2x2xf32>, %arg1: vector<2x4xf32>) {
  // expected-error@+1 {{'vector.shuffle' op dimension mismatch}}
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2x2xf32>, vector<2x4xf32>
}

// -----

func.func @shuffle_index_out_of_range(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
  // expected-error@+1 {{'vector.shuffle' op mask index #2 out of range}}
  %1 = vector.shuffle %arg0, %arg1 [0, 4] : vector<2xf32>, vector<2xf32>
}

// -----

func.func @shuffle_empty_mask(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
  // expected-error@+1 {{'vector.shuffle' op invalid mask length}}
  %1 = vector.shuffle %arg0, %arg1 [] : vector<2xf32>, vector<2xf32>
}

// -----

func.func @extract_element(%arg0: vector<f32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{expected position to be empty with 0-D vector}}
  %1 = vector.extractelement %arg0[%c : i32] : vector<f32>
}

// -----

func.func @extract_element(%arg0: vector<4xf32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{expected position for 1-D vector}}
  %1 = vector.extractelement %arg0[] : vector<4xf32>
}

// -----

func.func @extract_element(%arg0: vector<4x4xf32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{unexpected >1 vector rank}}
  %1 = vector.extractelement %arg0[%c : i32] : vector<4x4xf32>
}

// -----

func.func @extract_vector_type(%arg0: index) {
  // expected-error@+1 {{invalid kind of type specified}}
  %1 = vector.extract %arg0[] : index
}

// -----

func.func @extract_position_rank_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = vector.extract %arg0[0, 0, 0, 0] : vector<4x8x16xf32>
}

// -----

func.func @extract_position_rank_overflow_generic(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = "vector.extract" (%arg0) { position = [0, 0, 0, 0] } : (vector<4x8x16xf32>) -> (vector<16xf32>)
}

// -----

func.func @extract_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #2 to be a non-negative integer smaller than the corresponding vector dimension}}
  %1 = vector.extract %arg0[0, 43, 0] : vector<4x8x16xf32>
}

// -----

func.func @extract_precise_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a non-negative integer smaller than the corresponding vector dimension}}
  %1 = vector.extract %arg0[3, 7, 16] : vector<4x8x16xf32>
}

// -----

func.func @extract_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a non-negative integer smaller than the corresponding vector dimension}}
  %1 = vector.extract %arg0[0, 0, -1] : vector<4x8x16xf32>
}

// -----

func.func @insert_element(%arg0: f32, %arg1: vector<f32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{expected position to be empty with 0-D vector}}
  %0 = vector.insertelement %arg0, %arg1[%c : i32] : vector<f32>
}

// -----

func.func @insert_element(%arg0: f32, %arg1: vector<4xf32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{expected position for 1-D vector}}
  %0 = vector.insertelement %arg0, %arg1[] : vector<4xf32>
}

// -----

func.func @insert_element(%arg0: f32, %arg1: vector<4x4xf32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{unexpected >1 vector rank}}
  %0 = vector.insertelement %arg0, %arg1[%c : i32] : vector<4x4xf32>
}

// -----

func.func @insert_element_wrong_type(%arg0: i32, %arg1: vector<4xf32>) {
  %c = arith.constant 3 : i32
  // expected-error@+1 {{'vector.insertelement' op failed to verify that source operand type matches element type of result}}
  %0 = "vector.insertelement" (%arg0, %arg1, %c) : (i32, vector<4xf32>, i32) -> (vector<4xf32>)
}

// -----

func.func @insert_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than dest vector rank}}
  %1 = vector.insert %a, %b[3, 3, 3, 3, 3, 3] : f32 into vector<4x8x16xf32>
}

// -----

func.func @insert_vector_type(%a: vector<4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute rank + source rank to match dest vector rank}}
  %1 = vector.insert %a, %b[3] : vector<4xf32> into vector<4x8x16xf32>
}

// -----

func.func @insert_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute rank to match the dest vector rank}}
  %1 = vector.insert %a, %b[3, 3] : f32 into vector<4x8x16xf32>
}

// -----

func.func @insert_position_overflow(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a non-negative integer smaller than the corresponding dest vector dimension}}
  %1 = vector.insert %a, %b[0, 0, -1] : f32 into vector<4x8x16xf32>
}

// -----

func.func @insert_precise_position_overflow(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #1 to be a non-negative integer smaller than the corresponding dest vector dimension}}
  %1 = vector.insert %a, %b[4, 7, 15] : f32 into vector<4x8x16xf32>
}

// -----

func.func @outerproduct_num_operands(%arg0: f32) {
  // expected-error@+1 {{expected at least 2 operands}}
  %1 = vector.outerproduct %arg0 : f32, f32
}
// -----

func.func @outerproduct_non_vector_operand(%arg0: f32) {
  // expected-error@+1 {{expected vector type for operand #1}}
  %1 = vector.outerproduct %arg0, %arg0 : f32, f32
}

// -----

func.func @outerproduct_operand_1(%arg0: vector<4xf32>, %arg1: vector<4x8xf32>) {
  // expected-error@+1 {{expected 1-d vector for operand #1}}
  %1 = vector.outerproduct %arg1, %arg1 : vector<4x8xf32>, vector<4x8xf32>
}

// -----

func.func @outerproduct_operand_2(%arg0: vector<4xf32>, %arg1: vector<4x8xf32>) {
  // expected-error@+1 {{expected 1-d vector for operand #2}}
  %1 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<4x8xf32>
}

// -----

func.func @outerproduct_result_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected 2-d vector result}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<8xf32>)
}

// -----

func.func @outerproduct_operand_1_dim_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected #1 operand dim to match result dim #1}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<8x16xf32>)
}

// -----

func.func @outerproduct_operand_2_dim_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected #2 operand dim to match result dim #2}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<4x16xf32>)
}

// -----

func.func @outerproduct_axpy_operand(%arg0: vector<4x8xf32>, %arg1: f32) {
  // expected-error@+1 {{expected 1-d vector for operand #1}}
  %1 = vector.outerproduct %arg0, %arg1 : vector<4x8xf32>, f32
}

// -----

func.func @outerproduct_axpy_result_generic(%arg0: vector<4xf32>, %arg1: f32) {
  // expected-error@+1 {{expected 1-d vector result}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, f32) -> (vector<4x8xf32>)
}

// -----

func.func @outerproduct_axpy_operand_dim_generic(%arg0: vector<8xf32>, %arg1: f32) {
  // expected-error@+1 {{expected #1 operand dim to match result dim #1}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<8xf32>, f32) -> (vector<16xf32>)
}

// -----

func.func @outerproduct_operand_3_result_type_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x16xf32>) {
  // expected-error@+1 {{expected operand #3 of same type as result type}}
  %1 = "vector.outerproduct" (%arg0, %arg1, %arg2) : (vector<4xf32>, vector<8xf32>, vector<4x16xf32>) -> (vector<4x8xf32>)
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires two types}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst { permutation_map = affine_map<()->(0)> } : memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: vector<4x3xf32>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<4x3xf32>
  // expected-error@+1 {{ requires memref or ranked tensor type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0 : vector<4x3xf32>, vector<1x1x2x3xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<4x3xf32>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<4x3xf32>
  // expected-error@+1 {{ requires vector type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0 : memref<4x3xf32>, f32
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires 2 indices}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst { permutation_map = affine_map<()->(0)> } : memref<?x?xf32>, vector<128xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the source type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0)->(d0)>} : memref<?x?xf32>, vector<128xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : memref<?x?xf32>, vector<128xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d0 + d1)>} : memref<?x?xf32>, vector<128xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d0 + 1)>} : memref<?x?xf32>, vector<128xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst {permutation_map = affine_map<(d0, d1, d2)->(d0, d0)>} : memref<?x?x?xf32>, vector<3x7xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?x?xf32>) {
  %c1 = arith.constant 1 : i1
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-note@+1 {{prior use here}}
  %mask = vector.splat %c1 : vector<3x8x7xi1>
  // expected-error@+1 {{expects different type than prior uses: 'vector<3x7xi1>' vs 'vector<3x8x7xi1>'}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst, %mask {permutation_map = affine_map<(d0, d1, d2)->(d0, 0, d2)>} : memref<?x?x?xf32>, vector<3x8x7xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xvector<4x3xf32>>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<4x3xf32>
  // expected-error@+1 {{requires source vector element and vector result ranks to match}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0 {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : memref<?x?xvector<4x3xf32>>, vector<3xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xvector<6xf32>>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<6xf32>
  // expected-error@+1 {{requires the bitwidth of the minor 1-D vector to be an integral multiple of the bitwidth of the minor 1-D vector of the source}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0 : memref<?x?xvector<6xf32>>, vector<3xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xvector<2x3xf32>>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<2x3xf32>
  // expected-error@+1 {{ expects the optional in_bounds attr of same rank as permutation_map results: affine_map<(d0, d1) -> (d0, d1)>}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0 {in_bounds = [true], permutation_map = affine_map<(d0, d1)->(d0, d1)>} : memref<?x?xvector<2x3xf32>>, vector<1x1x2x3xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xvector<2x3xf32>>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<2x3xf32>
  // expected-error@+1 {{requires broadcast dimensions to be in-bounds}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0 {in_bounds = [false, true], permutation_map = affine_map<(d0, d1)->(0, d1)>} : memref<?x?xvector<2x3xf32>>, vector<1x1x2x3xf32>
}

// -----

func.func @test_vector.transfer_read(%arg0: memref<?x?xvector<2x3xf32>>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<2x3xf32>
  %mask = vector.splat %c1 : vector<2x3xi1>
  // expected-error@+1 {{does not support masks with vector element type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %vf0, %mask {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : memref<?x?xvector<2x3xf32>>, vector<1x1x2x3xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{requires two types}}
  vector.transfer_write %arg0, %arg0[%c3, %c3] : memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<vector<4x3xf32>>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<4x3xf32>
  // expected-error@+1 {{ requires vector type}}
  vector.transfer_write %arg0, %arg0[%c3, %c3] : memref<vector<4x3xf32>>, vector<4x3xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: vector<4x3xf32>) {
  %c3 = arith.constant 3 : index
  %f0 = arith.constant 0.0 : f32
  %vf0 = vector.splat %f0 : vector<4x3xf32>
  // expected-error@+1 {{ requires memref or ranked tensor type}}
  vector.transfer_write %arg0, %arg0[%c3, %c3] : vector<4x3xf32>, f32
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{expected 5 operand types but had 4}}
  %0 = "vector.transfer_write"(%cst, %arg0, %c3, %c3, %c3) {permutation_map = affine_map<()->(0)>} : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires 2 indices}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = affine_map<()->(0)>} : vector<128xf32>, memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the source type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = affine_map<(d0)->(d0)>} : vector<128xf32>, memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : vector<128xf32>, memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0 + d1)>} : vector<128xf32>, memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0 + 1)>} : vector<128xf32>, memref<?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?x?x?xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<3.0> : vector<3 x 7 x f32>
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = affine_map<(d0, d1, d2)->(d0, d0)>} : vector<3x7xf32>, memref<?x?x?xf32>
}

// -----

func.func @test_vector.transfer_write(%arg0: memref<?xf32>, %arg1: vector<7xf32>) {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  // expected-error@+1 {{should not have broadcast dimensions}}
  vector.transfer_write %arg1, %arg0[%c3]
      {permutation_map = affine_map<(d0) -> (0)>}
      : vector<7xf32>, memref<?xf32>
}

// -----

func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets of same size as destination vector rank}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [100], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected strides of same size as source vector rank}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 2, 2], strides = [1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected source rank to be smaller than destination rank}}
  %1 = vector.insert_strided_slice %b, %a {offsets = [2, 2], strides = [1, 1, 1]} : vector<4x8x16xf32> into vector<4x4xf32>
}

// -----

func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected offsets dimension 0 to be confined to [0, 4)}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [100,100,100], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected strides to be confined to [1, 2)}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 2, 2], strides = [100, 100]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected sum(offsets, source vector shape) dimension 1 to be confined to [1, 9)}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 7, 2], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets, sizes and strides attributes of same size}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [100], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets attribute of rank smaller than vector rank}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2, 2, 2, 2], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets attribute of rank smaller than vector rank}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2, 2, 2, 2], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected offsets dimension 0 to be confined to [0, 4)}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [100], sizes = [100], strides = [100]} : vector<4x8x16xf32> to vector<100x8x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected sizes dimension 0 to be confined to [1, 5)}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [100], strides = [100]} : vector<4x8x16xf32> to vector<100x8x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected strides to be confined to [1, 2)}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [1], strides = [100]} : vector<4x8x16xf32> to vector<1x8x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected strides to be confined to [1, 2)}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [1], strides = [100]} : vector<4x8x16xf32> to vector<1x8x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected sum(offsets, sizes) dimension 0 to be confined to [1, 5)}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [3], strides = [1]} : vector<4x8x16xf32> to vector<3x8x16xf32>
}

// -----

func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected result type to be 'vector<2x8x16xf32>'}}
  %1 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4x8x16xf32> to vector<3x1xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected an indexing map for each vector operand}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, c0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected indexing map 0 to be a projected permutation of its inputs}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1)[s0] -> (b0, s0, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{op expected indexing map 1 to have no symbols}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected indexing map 2 to have 5 number of inputs}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected indexing map 1 to have 4 number of outputs}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, b1, b2) -> (b1, b0, b2, f0)>,
  affine_map<(b0, f0, f1, b1, b2) -> (b0, b2, b1, f1)>,
  affine_map<(b0, f0, f1, b1, b2) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{op expected at least one contracting dimension pair}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c1, b0, c0, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{invalid contracting dimension map}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (f1, c1, c0, b0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{invalid batch dimension map}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<88x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{invalid accumulator/result vector shape}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<88x15x5xf32>
}

// -----

#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func.func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  %lhs_mask = vector.constant_mask [7, 8, 16, 15] : vector<7x8x16x15xi1>
  %rhs_mask = vector.constant_mask [8, 16, 7, 5] : vector<8x16x7x5xi1>
  // expected-error@+1 {{expected zero or exactly 2 vector mask operands}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2, %lhs_mask
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
}

// -----

#contraction_accesses = [
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
      ]
#contraction_trait = {
        indexing_maps = #contraction_accesses,
        iterator_types = ["parallel", "parallel", "reduction"]
      }
func.func @contraction(%arg0: vector<4x3xi32>,
                  %arg1: vector<3x7xf32>,
                  %arg2: vector<4x7xf32>) -> vector<4x7xf32> {
  // expected-error@+1 {{'vector.contract' op failed to verify that lhs and rhs have same element type}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
    : vector<4x3xi32>, vector<3x7xf32> into vector<4x7xf32>
}

// -----

#contraction_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}
func.func @contraction(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<2x3xf32>)
-> vector<3x2xf32>
{
// expected-error@+1 {{invalid accumulator/result vector shape, expected: 'vector<3x2xf32>'}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// -----

func.func @contract_with_dim_unused_by_lhs_and_rhs(%arg0 : vector<1x2xi32>, %arg1 : vector<2xi32>, %arg2 : vector<1xi32>) -> vector<1xi32> {
// expected-error@+1 {{'vector.contract' op expected all dimensions to be either a LHS or a RHS dimension}}
  %result = vector.contract {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2)>,
      affine_map<(d0, d1, d2) -> (d1)>
    ],
    iterator_types = ["reduction", "parallel", "reduction"],
    kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x2xi32>, vector<2xi32> into vector<1xi32>
  return  %result : vector<1xi32>
}

// -----

func.func @create_mask_0d_no_operands() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{must specify exactly one operand for 0-D create_mask}}
  %0 = vector.create_mask : vector<i1>
}

// -----

func.func @create_mask_0d_many_operands() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // expected-error@+1 {{must specify exactly one operand for 0-D create_mask}}
  %0 = vector.create_mask %c1, %c2, %c3 : vector<i1>
}

// -----

func.func @create_mask() {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // expected-error@+1 {{must specify an operand for each result vector dimension}}
  %0 = vector.create_mask %c3, %c2 : vector<4x3x7xi1>
}


// -----

func.func @constant_mask_0d_no_attr() {
  // expected-error@+1 {{array attr must have length 1 for 0-D vectors}}
  %0 = vector.constant_mask [] : vector<i1>
}

// -----

func.func @constant_mask_0d_bad_attr() {
  // expected-error@+1 {{mask dim size must be either 0 or 1 for 0-D vectors}}
  %0 = vector.constant_mask [2] : vector<i1>
}

// -----

func.func @constant_mask() {
  // expected-error@+1 {{must specify array attr of size equal vector result rank}}
  %0 = vector.constant_mask [3, 2, 7] : vector<4x3xi1>
}

// -----

func.func @constant_mask_out_of_bounds() {
  // expected-error@+1 {{array attr of size out of bounds of vector result dimension size}}
  %0 = vector.constant_mask [-1, 2] : vector<4x3xi1>
}

// -----

func.func @constant_mask_out_of_bounds() {
  // expected-error@+1 {{array attr of size out of bounds of vector result dimension size}}
  %0 = vector.constant_mask [3, 4] : vector<4x3xi1>
}

// -----

func.func @constant_mask_with_zero_mask_dim_size() {
  // expected-error@+1 {{expected all mask dim sizes to be zeros, as a result of conjunction with zero mask dim}}
  %0 = vector.constant_mask [0, 2] : vector<4x3xi1>
}

// -----

func.func @constant_mask_scalable_non_zero_dim_size() {
  // expected-error@+1 {{expected mask dim sizes for scalable masks to be 0}}
  %0 = vector.constant_mask [2] : vector<[8]xi1>
}

// -----

func.func @print_no_result(%arg0 : f32) -> i32 {
  // expected-error@+1 {{cannot name an operation with no results}}
  %0 = vector.print %arg0 : f32
}

// -----

func.func @reshape_bad_input_shape(%arg0 : vector<3x2x4xf32>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  // expected-error@+1 {{invalid input shape for vector type}}
  %1 = vector.reshape %arg0, [%c3, %c6, %c3], [%c2, %c9], [4]
    : vector<3x2x4xf32> to vector<2x3x4xf32>
}

// -----

func.func @reshape_bad_output_shape(%arg0 : vector<3x2x4xf32>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  // expected-error@+1 {{invalid output shape for vector type}}
  %1 = vector.reshape %arg0, [%c3, %c6], [%c2, %c9, %c3], [4]
    : vector<3x2x4xf32> to vector<2x3x4xf32>
}

// -----

func.func @reshape_bad_input_output_shape_product(%arg0 : vector<3x2x4xf32>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  // expected-error@+1 {{product of input and output shape sizes must match}}
  %1 = vector.reshape %arg0, [%c3, %c6], [%c2, %c6], [4]
    : vector<3x2x4xf32> to vector<2x3x4xf32>
}

// -----

func.func @reshape_bad_input_fixed_size(%arg0 : vector<3x2x5xf32>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  // expected-error@+1 {{fixed vector size must match input vector for dim 0}}
  %1 = vector.reshape %arg0, [%c3, %c6], [%c2, %c9], [4]
    : vector<3x2x5xf32> to vector<2x3x4xf32>
}

// -----

func.func @reshape_bad_output_fixed_size(%arg0 : vector<3x2x4xf32>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  // expected-error@+1 {{fixed vector size must match output vector for dim 0}}
  %1 = vector.reshape %arg0, [%c3, %c6], [%c2, %c9], [4]
    : vector<3x2x4xf32> to vector<2x3x5xf32>
}

// -----

func.func @shape_cast_wrong_element_type(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{op source/result vectors must have same element type}}
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<15x2xi32>
}

// -----

func.func @shape_cast_wrong_num_elements(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{op source/result number of elements must match}}
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<10x2xf32>
}

// -----

func.func @shape_cast_invalid_rank_reduction(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{invalid shape cast}}
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<2x15xf32>
}

// -----

func.func @shape_cast_invalid_rank_expansion(%arg0 : vector<15x2xf32>) {
  // expected-error@+1 {{invalid shape cast}}
  %0 = vector.shape_cast %arg0 : vector<15x2xf32> to vector<5x2x3x1xf32>
}

// -----

func.func @bitcast_not_vector(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{'vector.bitcast' invalid kind of type specified}}
  %0 = vector.bitcast %arg0 : vector<5x1x3x2xf32> to f32
}

// -----

func.func @bitcast_rank_mismatch_to_0d(%arg0 : vector<1xf32>) {
  // expected-error@+1 {{op failed to verify that all of {source, result} have same rank}}
  %0 = vector.bitcast %arg0 : vector<1xf32> to vector<f32>
}

// -----

func.func @bitcast_rank_mismatch_from_0d(%arg0 : vector<f32>) {
  // expected-error@+1 {{op failed to verify that all of {source, result} have same rank}}
  %0 = vector.bitcast %arg0 : vector<f32> to vector<1xf32>
}

// -----

func.func @bitcast_rank_mismatch(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{op failed to verify that all of {source, result} have same rank}}
  %0 = vector.bitcast %arg0 : vector<5x1x3x2xf32> to vector<5x3x2xf32>
}

// -----

func.func @bitcast_shape_mismatch(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{op dimension size mismatch}}
  %0 = vector.bitcast %arg0 : vector<5x1x3x2xf32> to vector<5x2x3x2xf32>
}

// -----

func.func @bitcast_sizemismatch(%arg0 : vector<5x1x3x2xf32>) {
  // expected-error@+1 {{op source/result bitwidth of the minor 1-D vectors must be equal}}
  %0 = vector.bitcast %arg0 : vector<5x1x3x2xf32> to vector<5x1x3x3xf16>
}

// -----

func.func @reduce_unknown_kind(%arg0: vector<16xf32>) -> f32 {
  // expected-error@+1 {{custom op 'vector.reduction' Unknown combining kind: joho}}
  %0 = vector.reduction <joho>, %arg0 : vector<16xf32> into f32
}

// -----

func.func @reduce_elt_type_mismatch(%arg0: vector<16xf32>) -> i32 {
  // expected-error@+1 {{'vector.reduction' op failed to verify that source operand and result have same element type}}
  %0 = vector.reduction <add>, %arg0 : vector<16xf32> into i32
}

// -----

func.func @reduce_unsupported_attr(%arg0: vector<16xf32>) -> i32 {
  // expected-error@+1 {{expected '<'}}
  %0 = vector.reduction 1234, %arg0 : vector<16xf32> into i32
}

// -----

func.func @reduce_unsupported_third_argument(%arg0: vector<16xf32>, %arg1: f32) -> f32 {
  // expected-error@+1 {{'vector.reduction' unsupported number of operands}}
  %0 = vector.reduction <add>, %arg0, %arg1, %arg1 : vector<16xf32> into f32
}

// -----

func.func @reduce_unsupported_rank(%arg0: vector<4x16xf32>) -> f32 {
  // expected-error@+1 {{'vector.reduction' op unsupported reduction rank: 2}}
  %0 = vector.reduction <add>, %arg0 : vector<4x16xf32> into f32
}

// -----

func.func @multi_reduce_invalid_type(%arg0: vector<4x16xf32>, %acc: vector<16xf32>) -> f32 {
  // expected-error@+1 {{'vector.multi_reduction' op destination type 'vector<16xf32>' is incompatible with source type 'vector<4x16xf32>'}}
  %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<4x16xf32> to vector<16xf32>
}

// -----

func.func @transpose_rank_mismatch(%arg0: vector<4x16x11xf32>) {
  // expected-error@+1 {{'vector.transpose' op vector result rank mismatch: 1}}
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<4x16x11xf32> to vector<100xf32>
}

// -----

func.func @transpose_length_mismatch(%arg0: vector<4x4xf32>) {
  // expected-error@+1 {{'vector.transpose' op transposition length mismatch: 3}}
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<4x4xf32> to vector<4x4xf32>
}

// -----

func.func @transpose_index_oob(%arg0: vector<4x4xf32>) {
  // expected-error@+1 {{'vector.transpose' op transposition index out of range: 2}}
  %0 = vector.transpose %arg0, [2, 0] : vector<4x4xf32> to vector<4x4xf32>
}

// -----

func.func @transpose_index_dup(%arg0: vector<4x4xf32>) {
  // expected-error@+1 {{'vector.transpose' op duplicate position index: 0}}
  %0 = vector.transpose %arg0, [0, 0] : vector<4x4xf32> to vector<4x4xf32>
}

// -----

func.func @transpose_dim_size_mismatch(%arg0: vector<11x7x3x2xi32>) {
  // expected-error@+1 {{'vector.transpose' op dimension size mismatch at: 0}}
  %0 = vector.transpose %arg0, [3, 0, 1, 2] : vector<11x7x3x2xi32> to vector<2x3x7x11xi32>
}

// -----

func.func @flat_transpose_type_mismatch(%arg0: vector<16xf32>) {
  // expected-error@+1 {{'vector.flat_transpose' op failed to verify that source operand and result have same element type}}
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 } : vector<16xf32> -> vector<16xf64>
}

// -----

func.func @type_cast_layout(%arg0: memref<4x3xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + s2)>>) {
  // expected-error@+1 {{expects operand to be a memref with identity layout}}
  %0 = vector.type_cast %arg0: memref<4x3xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + s2)>> to memref<vector<4x3xf32>>
}

// -----

func.func @store_unsupported_layout(%memref : memref<200x100xf32, affine_map<(d0, d1) -> (200*d0 + 2*d1)>>,
                               %i : index, %j : index, %value : vector<8xf32>) {
  // expected-error@+1 {{'vector.store' op most minor memref dim must have unit stride}}
  vector.store %value, %memref[%i, %j] : memref<200x100xf32, affine_map<(d0, d1) -> (200*d0 + 2*d1)>>,
                                         vector<8xf32>
  return
}

// -----

func.func @vector_memref_mismatch(%memref : memref<200x100xvector<4xf32>>, %i : index,
                             %j : index, %value : vector<8xf32>) {
  // expected-error@+1 {{'vector.store' op base memref and valueToStore vector types should match}}
  vector.store %value, %memref[%i, %j] : memref<200x100xvector<4xf32>>, vector<8xf32>
}

// -----

func.func @store_base_type_mismatch(%base : memref<?xf64>, %value : vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.store' op base and valueToStore element type should match}}
  vector.store %value, %base[%c0] : memref<?xf64>, vector<16xf32>
}

// -----

func.func @store_memref_index_mismatch(%base : memref<?xf32>, %value : vector<16xf32>) {
  // expected-error@+1 {{'vector.store' op requires 1 indices}}
  vector.store %value, %base[] : memref<?xf32>, vector<16xf32>
}

// -----

func.func @maskedload_base_type_mismatch(%base: memref<?xf64>, %mask: vector<16xi1>, %pass: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.maskedload' op base and result element type should match}}
  %0 = vector.maskedload %base[%c0], %mask, %pass : memref<?xf64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @maskedload_dim_mask_mismatch(%base: memref<?xf32>, %mask: vector<15xi1>, %pass: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.maskedload' op expected result dim to match mask dim}}
  %0 = vector.maskedload %base[%c0], %mask, %pass : memref<?xf32>, vector<15xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @maskedload_pass_thru_type_mask_mismatch(%base: memref<?xf32>, %mask: vector<16xi1>, %pass: vector<16xi32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.maskedload' op expected pass_thru of same type as result type}}
  %0 = vector.maskedload %base[%c0], %mask, %pass : memref<?xf32>, vector<16xi1>, vector<16xi32> into vector<16xf32>
}

// -----

func.func @maskedload_memref_mismatch(%base: memref<?xf32>, %mask: vector<16xi1>, %pass: vector<16xf32>) {
  // expected-error@+1 {{'vector.maskedload' op requires 1 indices}}
  %0 = vector.maskedload %base[], %mask, %pass : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @maskedstore_base_type_mismatch(%base: memref<?xf64>, %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.maskedstore' op base and valueToStore element type should match}}
  vector.maskedstore %base[%c0], %mask, %value : memref<?xf64>, vector<16xi1>, vector<16xf32>
}

// -----

func.func @maskedstore_dim_mask_mismatch(%base: memref<?xf32>, %mask: vector<15xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.maskedstore' op expected valueToStore dim to match mask dim}}
  vector.maskedstore %base[%c0], %mask, %value : memref<?xf32>, vector<15xi1>, vector<16xf32>
}

// -----

func.func @maskedstore_memref_mismatch(%base: memref<?xf32>, %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.maskedstore' op requires 1 indices}}
  vector.maskedstore %base[%c0, %c0], %mask, %value : memref<?xf32>, vector<16xi1>, vector<16xf32>
}

// -----

func.func @gather_base_type_mismatch(%base: memref<?xf64>, %indices: vector<16xi32>,
                                %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.gather' op base and result element type should match}}
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<?xf64>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @gather_memref_mismatch(%base: memref<?x?xf64>, %indices: vector<16xi32>,
                             %mask: vector<16xi1>, %pass_thru: vector<16xf64>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.gather' op requires 2 indices}}
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<?x?xf64>, vector<16xi32>, vector<16xi1>, vector<16xf64> into vector<16xf64>
}

// -----

func.func @gather_rank_mismatch(%base: memref<?xf32>, %indices: vector<16xi32>,
                           %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.gather' op result #0 must be  of ranks 1, but got 'vector<2x16xf32>'}}
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<2x16xf32>
}

// -----

func.func @gather_dim_indices_mismatch(%base: memref<?xf32>, %indices: vector<17xi32>,
                                  %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.gather' op expected result dim to match indices dim}}
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<?xf32>, vector<17xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @gather_dim_mask_mismatch(%base: memref<?xf32>, %indices: vector<16xi32>,
                               %mask: vector<17xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.gather' op expected result dim to match mask dim}}
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<?xf32>, vector<16xi32>, vector<17xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @gather_pass_thru_type_mismatch(%base: memref<?xf32>, %indices: vector<16xi32>,
                                     %mask: vector<16xi1>, %pass_thru: vector<16xf64>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.gather' op expected pass_thru of same type as result type}}
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf64> into vector<16xf32>
}

// -----

func.func @scatter_base_type_mismatch(%base: memref<?xf64>, %indices: vector<16xi32>,
                                 %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.scatter' op base and valueToStore element type should match}}
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<?xf64>, vector<16xi32>, vector<16xi1>, vector<16xf32>
}

// -----

func.func @scatter_memref_mismatch(%base: memref<?x?xf64>, %indices: vector<16xi32>,
                              %mask: vector<16xi1>, %value: vector<16xf64>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.scatter' op requires 2 indices}}
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<?x?xf64>, vector<16xi32>, vector<16xi1>, vector<16xf64>
}

// -----

func.func @scatter_rank_mismatch(%base: memref<?xf32>, %indices: vector<16xi32>,
                            %mask: vector<16xi1>, %value: vector<2x16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.scatter' op operand #4 must be  of ranks 1, but got 'vector<2x16xf32>'}}
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<2x16xf32>
}

// -----

func.func @scatter_dim_indices_mismatch(%base: memref<?xf32>, %indices: vector<17xi32>,
                                   %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.scatter' op expected valueToStore dim to match indices dim}}
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<?xf32>, vector<17xi32>, vector<16xi1>, vector<16xf32>
}

// -----

func.func @scatter_dim_mask_mismatch(%base: memref<?xf32>, %indices: vector<16xi32>,
                                %mask: vector<17xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.scatter' op expected valueToStore dim to match mask dim}}
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<?xf32>, vector<16xi32>, vector<17xi1>, vector<16xf32>
}

// -----

func.func @expand_base_type_mismatch(%base: memref<?xf64>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.expandload' op base and result element type should match}}
  %0 = vector.expandload %base[%c0], %mask, %pass_thru : memref<?xf64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @expand_dim_mask_mismatch(%base: memref<?xf32>, %mask: vector<17xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.expandload' op expected result dim to match mask dim}}
  %0 = vector.expandload %base[%c0], %mask, %pass_thru : memref<?xf32>, vector<17xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @expand_pass_thru_mismatch(%base: memref<?xf32>, %mask: vector<16xi1>, %pass_thru: vector<17xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.expandload' op expected pass_thru of same type as result type}}
  %0 = vector.expandload %base[%c0], %mask, %pass_thru : memref<?xf32>, vector<16xi1>, vector<17xf32> into vector<16xf32>
}

// -----

func.func @expand_memref_mismatch(%base: memref<?x?xf32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.expandload' op requires 2 indices}}
  %0 = vector.expandload %base[%c0], %mask, %pass_thru : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
}

// -----

func.func @compress_base_type_mismatch(%base: memref<?xf64>, %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.compressstore' op base and valueToStore element type should match}}
  vector.compressstore %base[%c0], %mask, %value : memref<?xf64>, vector<16xi1>, vector<16xf32>
}

// -----

func.func @compress_dim_mask_mismatch(%base: memref<?xf32>, %mask: vector<17xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.compressstore' op expected valueToStore dim to match mask dim}}
  vector.compressstore %base[%c0], %mask, %value : memref<?xf32>, vector<17xi1>, vector<16xf32>
}

// -----

func.func @compress_memref_mismatch(%base: memref<?x?xf32>, %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'vector.compressstore' op requires 2 indices}}
  vector.compressstore %base[%c0, %c0, %c0], %mask, %value : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
}

// -----

func.func @extract_map_rank(%v: vector<32xf32>, %id : index) {
  // expected-error@+1 {{'vector.extract_map' op expected source and destination vectors of same rank}}
  %0 = vector.extract_map %v[%id] : vector<32xf32> to vector<2x1xf32>
}

// -----

func.func @extract_map_size(%v: vector<63xf32>, %id : index) {
  // expected-error@+1 {{'vector.extract_map' op source vector dimensions must be a multiple of destination vector dimensions}}
  %0 = vector.extract_map %v[%id] : vector<63xf32> to vector<2xf32>
}

// -----

func.func @extract_map_id(%v: vector<2x32xf32>, %id : index) {
  // expected-error@+1 {{'vector.extract_map' op expected number of ids must match the number of dimensions distributed}}
  %0 = vector.extract_map %v[%id] : vector<2x32xf32> to vector<1x1xf32>
}

// -----

func.func @insert_map_rank(%v: vector<2x1xf32>, %v1: vector<32xf32>, %id : index) {
  // expected-error@+1 {{'vector.insert_map' op expected source and destination vectors of same rank}}
  %0 = vector.insert_map %v, %v1[%id] : vector<2x1xf32> into vector<32xf32>
}

// -----

func.func @insert_map_size(%v: vector<3xf32>, %v1: vector<64xf32>, %id : index) {
  // expected-error@+1 {{'vector.insert_map' op destination vector size must be a multiple of source vector size}}
  %0 = vector.insert_map %v, %v1[%id] : vector<3xf32> into vector<64xf32>
}

// -----

func.func @insert_map_id(%v: vector<2x1xf32>, %v1: vector<4x32xf32>, %id : index) {
  // expected-error@+1 {{'vector.insert_map' op expected number of ids must match the number of dimensions distributed}}
  %0 = vector.insert_map %v, %v1[%id] : vector<2x1xf32> into vector<4x32xf32>
}

// -----

func.func @scan_reduction_dim_constraint(%arg0: vector<2x3xi32>, %arg1: vector<3xi32>) -> vector<3xi32> {
  // expected-error@+1 {{'vector.scan' op reduction dimension 5 has to be less than 2}}
  %0:2 = vector.scan <add>, %arg0, %arg1 {inclusive = true, reduction_dim = 5} :
    vector<2x3xi32>, vector<3xi32>
  return %0#1 : vector<3xi32>
}

// -----

func.func @scan_ival_rank_constraint(%arg0: vector<2x3xi32>, %arg1: vector<1x3xi32>) -> vector<1x3xi32> {
  // expected-error@+1 {{initial value rank 2 has to be equal to 1}}
  %0:2 = vector.scan <add>, %arg0, %arg1 {inclusive = true, reduction_dim = 0} :
    vector<2x3xi32>, vector<1x3xi32>
  return %0#1 : vector<1x3xi32>
}

// -----

func.func @scan_incompatible_shapes(%arg0: vector<2x3xi32>, %arg1: vector<5xi32>) -> vector<2x3xi32> {
  // expected-error@+1 {{incompatible input/initial value shapes}}
  %0:2 = vector.scan <add>, %arg0, %arg1 {inclusive = true, reduction_dim = 0} :
    vector<2x3xi32>, vector<5xi32>
  return %0#0 : vector<2x3xi32>
}

// -----

func.func @scan_unsupported_kind(%arg0: vector<2x3xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  // expected-error@+1 {{'vector.scan' op unsupported reduction type 'f32' for kind 'xor'}}
  %0:2 = vector.scan <xor>, %arg0, %arg1 {inclusive = true, reduction_dim = 0} :
    vector<2x3xf32>, vector<3xf32>
  return %0#0 : vector<2x3xf32>
}

// -----

func.func @invalid_splat(%v : f32) {
  // expected-error@+1 {{invalid kind of type specified}}
  vector.splat %v : memref<8xf32>
  return
}

// -----

func.func @warp_wrong_num_outputs(%laneid: index) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op expected same number of yield operands and return values.}}
  %2 = vector.warp_execute_on_lane_0(%laneid)[64] -> (vector<4xi32>) {
  }
  return
}

// -----

func.func @warp_wrong_num_inputs(%laneid: index) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op expected same number op arguments and block arguments.}}
  vector.warp_execute_on_lane_0(%laneid)[64] {
  ^bb0(%arg0 : vector<128xi32>) :
  }
  return
}

// -----

func.func @warp_wrong_return_distribution(%laneid: index) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op incompatible distribution dimensions from 'vector<128xi32>' to 'vector<4xi32>'}}
  %2 = vector.warp_execute_on_lane_0(%laneid)[64] -> (vector<4xi32>) {
    %0 = arith.constant dense<2>: vector<128xi32>
    vector.yield %0 : vector<128xi32>
  }
  return
}


// -----

func.func @warp_wrong_arg_distribution(%laneid: index, %v0 : vector<4xi32>) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op incompatible distribution dimensions from 'vector<128xi32>' to 'vector<4xi32>'}}
  vector.warp_execute_on_lane_0(%laneid)[64]
  args(%v0 : vector<4xi32>) {
   ^bb0(%arg0 : vector<128xi32>) :
  }
  return
}

// -----

func.func @warp_2_distributed_dims(%laneid: index) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op expected only one dimension to be distributed from 'vector<128x128xi32>' to 'vector<4x4xi32>'}}
  %2 = vector.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x4xi32>) {
    %0 = arith.constant dense<2>: vector<128x128xi32>
    vector.yield %0 : vector<128x128xi32>
  }
  return
}

// -----

func.func @warp_mismatch_rank(%laneid: index) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op expected distributed vectors to have same rank and element type.}}
  %2 = vector.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x4xi32>) {
    %0 = arith.constant dense<2>: vector<128xi32>
    vector.yield %0 : vector<128xi32>
  }
  return
}

// -----

func.func @warp_mismatch_rank(%laneid: index) {
  // expected-error@+1 {{'vector.warp_execute_on_lane_0' op expected vector type for distributed operands.}}
  %2 = vector.warp_execute_on_lane_0(%laneid)[32] -> (i32) {
    %0 = arith.constant dense<2>: vector<128xi32>
    vector.yield %0 : vector<128xi32>
  }
  return
}
