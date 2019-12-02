// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @broadcast_rank_too_high(%arg0: vector<4x4xf32>) {
  // expected-error@+1 {{source rank higher than destination rank}}
  %1 = vector.broadcast %arg0 : vector<4x4xf32> to vector<4xf32>
}

// -----

func @broadcast_dim1_mismatch(%arg0: vector<7xf32>) {
  // expected-error@+1 {{vector.broadcast' op dimension mismatch (7 vs. 3)}}
  %1 = vector.broadcast %arg0 : vector<7xf32> to vector<3xf32>
}

// -----

func @broadcast_dim2_mismatch(%arg0: vector<4x8xf32>) {
  // expected-error@+1 {{vector.broadcast' op dimension mismatch (4 vs. 1)}}
  %1 = vector.broadcast %arg0 : vector<4x8xf32> to vector<1x8xf32>
}

// -----

func @extract_element_vector_type(%arg0: index) {
  // expected-error@+1 {{expected vector type}}
  %1 = vector.extractelement %arg0[] : index
}

// -----

func @extractelement_position_empty(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected non-empty position attribute}}
  %1 = vector.extractelement %arg0[] : vector<4x8x16xf32>
}

// -----

func @extractelement_position_rank_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = vector.extractelement %arg0[0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<4x8x16xf32>
}

// -----

func @extractelement_position_rank_overflow_generic(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = "vector.extractelement" (%arg0) { position = [0 : i32, 0 : i32, 0 : i32, 0 : i32] } : (vector<4x8x16xf32>) -> (vector<16xf32>)
}

// -----

func @extractelement_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #2 to be a non-negative integer smaller than the corresponding vector dimension}}
  %1 = vector.extractelement %arg0[0 : i32, 43 : i32, 0 : i32] : vector<4x8x16xf32>
}

// -----

func @extractelement_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a non-negative integer smaller than the corresponding vector dimension}}
  %1 = vector.extractelement %arg0[0 : i32, 0 : i32, -1 : i32] : vector<4x8x16xf32>
}

// -----

func @insert_element_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected non-empty position attribute}}
  %1 = vector.insertelement %a, %b[] : f32 into vector<4x8x16xf32>
}

// -----

func @insert_element_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than dest vector rank}}
  %1 = vector.insertelement %a, %b[3 : i32,3 : i32,3 : i32,3 : i32,3 : i32,3 : i32] : f32 into vector<4x8x16xf32>
}

// -----

func @insert_element_vector_type(%a: vector<4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute rank + source rank to match dest vector rank}}
  %1 = vector.insertelement %a, %b[3 : i32] : vector<4xf32> into vector<4x8x16xf32>
}

// -----

func @insert_element_vector_type(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute rank to match the dest vector rank}}
  %1 = vector.insertelement %a, %b[3 : i32,3 : i32] : f32 into vector<4x8x16xf32>
}

// -----

func @insertelement_position_overflow(%a: f32, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a non-negative integer smaller than the corresponding dest vector dimension}}
  %1 = vector.insertelement %a, %b[0 : i32, 0 : i32, -1 : i32] : f32 into vector<4x8x16xf32>
}

// -----

func @outerproduct_num_operands(%arg0: f32) {
  // expected-error@+1 {{expected at least 2 operands}}
  %1 = vector.outerproduct %arg0 : f32, f32
}
// -----

func @outerproduct_non_vector_operand(%arg0: f32) {
  // expected-error@+1 {{expected 2 vector types}}
  %1 = vector.outerproduct %arg0, %arg0 : f32, f32
}

// -----

func @outerproduct_operand_1(%arg0: vector<4xf32>, %arg1: vector<4x8xf32>) {
  // expected-error@+1 {{expected 1-d vector for operand #1}}
  %1 = vector.outerproduct %arg1, %arg1 : vector<4x8xf32>, vector<4x8xf32>
}

// -----

func @outerproduct_operand_2(%arg0: vector<4xf32>, %arg1: vector<4x8xf32>) {
  // expected-error@+1 {{expected 1-d vector for operand #2}}
  %1 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<4x8xf32>
}

// -----

func @outerproduct_result_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected 2-d vector result}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<8xf32>)
}

// -----

func @outerproduct_operand_1_dim_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected #1 operand dim to match result dim #1}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<8x16xf32>)
}

// -----

func @outerproduct_operand_2_dim_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected #2 operand dim to match result dim #2}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<4x16xf32>)
}

// -----

func @outerproduct_operand_3_result_type_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x16xf32>) {
  // expected-error@+1 {{expected operand #3 of same type as result type}}
  %1 = "vector.outerproduct" (%arg0, %arg1, %arg2) : (vector<4xf32>, vector<8xf32>, vector<4x16xf32>) -> (vector<4x8xf32>)
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{two types required}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst { permutation_map = ()->(0) } : memref<?x?xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires 2 indices}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst { permutation_map = ()->(0) } : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires attribute 'permutation_map'}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {perm = (d0)->(d0)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0)->(d0)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0, d1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0 + d1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0 + 1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst {permutation_map = (d0, d1, d2)->(d0, d0)} : memref<?x?x?xf32>, vector<3x7xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{expected 5 operand types but had 4}}
  %0 = "vector.transfer_write"(%cst, %arg0, %c3, %c3, %c3) {permutation_map = ()->(0)} : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires 2 indices}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = ()->(0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires attribute 'permutation_map'}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {perm = (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0, d1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + d1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + 1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<3 x 7 x f32>
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = (d0, d1, d2)->(d0, d0)} : vector<3x7xf32>, memref<?x?x?xf32>
}

// -----

func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets of same size as destination vector rank}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [100], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected strides of same size as source vector rank}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 2, 2], strides = [1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected source rank to be smaller than destination rank}}
  %1 = vector.insert_strided_slice %b, %a {offsets = [2, 2], strides = [1, 1, 1]} : vector<4x8x16xf32> into vector<4x4xf32>
}

// -----

func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected offsets dimension 0 to be confined to [0, 4)}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [100,100,100], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected strides to be confined to [1, 2)}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 2, 2], strides = [100, 100]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected sum(offsets, source vector shape) dimension 1 to be confined to [1, 9)}}
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 7, 2], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets, sizes and strides attributes of same size}}
  %1 = vector.strided_slice %arg0 {offsets = [100], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets attribute of rank smaller than vector rank}}
  %1 = vector.strided_slice %arg0 {offsets = [2, 2, 2, 2], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected offsets attribute of rank smaller than vector rank}}
  %1 = vector.strided_slice %arg0 {offsets = [2, 2, 2, 2], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected offsets dimension 0 to be confined to [0, 4)}}
  %1 = vector.strided_slice %arg0 {offsets = [100], sizes = [100], strides = [100]} : vector<4x8x16xf32> to vector<100x8x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected sizes dimension 0 to be confined to [1, 5)}}
  %1 = vector.strided_slice %arg0 {offsets = [2], sizes = [100], strides = [100]} : vector<4x8x16xf32> to vector<100x8x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected strides to be confined to [1, 2)}}
  %1 = vector.strided_slice %arg0 {offsets = [2], sizes = [1], strides = [100]} : vector<4x8x16xf32> to vector<1x8x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected strides to be confined to [1, 2)}}
  %1 = vector.strided_slice %arg0 {offsets = [2], sizes = [1], strides = [100]} : vector<4x8x16xf32> to vector<1x8x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected sum(offsets, sizes) dimension 0 to be confined to [1, 5)}}
  %1 = vector.strided_slice %arg0 {offsets = [2], sizes = [3], strides = [1]} : vector<4x8x16xf32> to vector<3x8x16xf32>
}

// -----

func @strided_slice(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{op expected result type to be 'vector<2x8x16xf32>'}}
  %1 = vector.strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4x8x16xf32> to vector<3x1xf32>
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected an indexing map for each vector operand}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, c0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected indexing map 0 to be a projected permutation of its inputs}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1)[s0] -> (b0, s0, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{op expected indexing map 1 to have no symbols}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected indexing map 2 to have 5 number of inputs}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{expected indexing map 1 to have 4 number of outputs}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, b1, b2) -> (b1, b0, b2, f0),
  (b0, f0, f1, b1, b2) -> (b0, b2, b1, f1),
  (b0, f0, f1, b1, b2) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{op expected at least one contracting dimension pair}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c1, b0, c0, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{invalid contracting dimension map}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (f1, c1, c0, b0),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{invalid batch dimension map}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<88x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // expected-error@+1 {{invalid accumulator/result vector shape}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<88x15x5xf32>
  return
}

// -----

#contraction_accesses = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
func @contraction(%arg0: vector<7x8x16x15xf32>, %arg1: vector<8x16x7x5xf32>,
                  %arg2: vector<8x15x5xf32>, %arg3 :  vector<8x15x8x5xf32>,
                  %arg4 : index) {
  %lhs_mask = vector.make_index_tuple %arg4, %arg4, %arg4, %arg4
    : tuple<index, index, index, index>
  %rhs_mask = vector.make_index_tuple %arg4, %arg4, %arg4, %arg4
    : tuple<index, index, index, index>
  // expected-error@+1 {{expected zero or exactly 2 vector mask operands}}
  %0 = vector.contract #contraction_trait %arg0, %arg1, %arg2, %lhs_mask
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  return
}
