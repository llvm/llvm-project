// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: contraction_dot
func.func @contraction_dot(%A: memref<1584xf32>, %B: memref<1584xf32>, %C: memref<f32>) {

// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [0] : vector<1584xf32> to f32
  linalg.dot ins(%A, %B: memref<1584xf32>, memref<1584xf32>)
            outs(%C: memref<f32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.dot"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
}

// -----

// CHECK-LABEL: contraction_matvec
func.func @contraction_matvec(%A: memref<1584x1584xf32>, %B: memref<1584xf32>, %C: memref<1584xf32>) {

// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584x1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [1] : vector<1584x1584xf32> to vector<1584xf32>
  linalg.matvec ins(%A, %B: memref<1584x1584xf32>, memref<1584xf32>)
            outs(%C: memref<1584xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matvec"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
}

// -----

// CHECK-LABEL: contraction_matmul
func.func @contraction_matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584x1584x1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [2] : vector<1584x1584x1584xf32> to vector<1584x1584xf32>
  linalg.matmul ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
            outs(%C: memref<1584x1584xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
}

// -----

// CHECK-LABEL: contraction_batch_matmul
func.func @contraction_batch_matmul(%A: memref<1584x1584x1584xf32>, %B: memref<1584x1584x1584xf32>, %C: memref<1584x1584x1584xf32>) {
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584x1584x1584x1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [3] : vector<1584x1584x1584x1584xf32> to vector<1584x1584x1584xf32>
  linalg.batch_matmul
    ins(%A, %B: memref<1584x1584x1584xf32>, memref<1584x1584x1584xf32>)
   outs(%C: memref<1584x1584x1584xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.batch_matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
}

// -----

#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @vectorization_test
func.func @vectorization_test(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x32x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<8x32x16xf32>
  //       CHECK: %[[ACC:.*]] = vector.transfer_read %{{.*}} : memref<8x32xf32>, vector<8x32xf32>
  //       CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8x32x16xf32>
  //       CHECK: %[[R:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[ACC]] [2] : vector<8x32x16xf32> to vector<8x32xf32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xf32>, memref<8x32xf32>
  linalg.generic #matmul_trait
    ins(%A, %B : memref<8x16xf32>, memref<16x32xf32>)
   outs(%C : memref<8x32xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

#matmul_transpose_out_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (n, m)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @generic_output_transpose
func.func @generic_output_transpose(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                                    %C: memref<32x8xf32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x32x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<8x32x16xf32>
  //       CHECK: %[[ACC:.*]] = vector.transfer_read %{{.*}} : memref<32x8xf32>, vector<8x32xf32>
  //       CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8x32x16xf32>
  //       CHECK: %[[R:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[ACC]] [2] : vector<8x32x16xf32> to vector<8x32xf32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xf32>, memref<32x8xf32>
  linalg.generic #matmul_transpose_out_trait
    ins(%A, %B : memref<8x16xf32>, memref<16x32xf32>)
   outs(%C : memref<32x8xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: func @generic_interchanged_transpose
func.func @generic_interchanged_transpose(%arg0: tensor<12x128x32xf32>) -> tensor<128x12x32xf32> {
  // CHECK: %[[IN:.+]] = vector.transfer_read
  // CHECK: vector.transfer_write %[[IN]], {{.+}} permutation_map = #[[MAP]]
  %0 = tensor.empty() : tensor<128x12x32xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1],
                       iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<12x128x32xf32>)
    outs(%0 : tensor<128x12x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<128x12x32xf32>
  return %1 : tensor<128x12x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @vectorization_test_integer
func.func @vectorization_test_integer(%A: memref<8x16xi32>, %B: memref<16x32xi32>,
                                 %C: memref<8x32xi32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xi32>, vector<8x32x16xi32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xi32>, vector<8x32x16xi32>
  //       CHECK: %[[ACC:.*]] = vector.transfer_read %{{.*}} : memref<8x32xi32>, vector<8x32xi32>
  //       CHECK: %[[MUL:.*]] = arith.muli %{{.*}}, %{{.*}} : vector<8x32x16xi32>
  //       CHECK: vector.multi_reduction <add>, %[[MUL]], %[[ACC]] [2] : vector<8x32x16xi32> to vector<8x32xi32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xi32>, memref<8x32xi32>
  linalg.generic #matmul_trait
    ins(%A, %B : memref<8x16xi32>, memref<16x32xi32>)
   outs(%C : memref<8x32xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32) :
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

// CHECK-LABEL: func @vectorization_test_2
func.func @vectorization_test_2(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<8x32x16xf32>
  //       CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [2] : vector<8x32x16xf32> to vector<8x32xf32>
  linalg.matmul
    ins(%A, %B: memref<8x16xf32>, memref<16x32xf32>)
   outs(%C: memref<8x32xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
}

// -----

// CHECK-LABEL: func @test_vectorize_scalar_input
func.func @test_vectorize_scalar_input(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.generic {
    indexing_maps = [affine_map<(m, n) -> ()>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
   ins(%arg0 : f32)
  outs(%A: memref<8x16xf32>) {
    ^bb(%0: f32, %1: f32) :
      linalg.yield %0 : f32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_do_not_vectorize_unsupported_element_types
func.func @test_do_not_vectorize_unsupported_element_types(%A : memref<8x16xcomplex<f32>>, %arg0 : complex<f32>) {
  // CHECK-NOT: vector.broadcast
  // CHECK-NOT: vector.transfer_write
  linalg.generic {
    indexing_maps = [affine_map<(m, n) -> ()>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
   ins(%arg0 : complex<f32>)
  outs(%A: memref<8x16xcomplex<f32>>) {
    ^bb(%0: complex<f32>, %1: complex<f32>) :
      linalg.yield %0 : complex<f32>
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

#map0 = affine_map<(d0) -> (d0)>

func.func @vectorize_affine_apply(%arg0: tensor<32xf32>, %arg3: index) -> tensor<32xi32> {
  %0 = tensor.empty() : tensor<32xi32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0],
                       iterator_types = ["parallel"]}
    ins(%arg0 : tensor<32xf32>)
    outs(%0 : tensor<32xi32>) {
  ^bb0(%arg1: f32, %arg2: i32):
    %2 = linalg.index 0 : index
    %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %arg3)
    %13 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%12)[%arg3]
    %3 = arith.index_cast %13 : index to i32
    linalg.yield %3 : i32
  } -> tensor<32xi32>
  return %1 : tensor<32xi32>
}

// CHECK-LABEL:  func.func @vectorize_affine_apply
// CHECK-SAME: %arg0: tensor<32xf32>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK:   %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<32xindex>
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[EMPTY:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:   %[[BCAST:.*]] = vector.broadcast %[[ARG1]] : index to vector<32xindex>
// CHECK:   %[[ADDI:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<32xindex>
// CHECK:   %[[BCAST2:.*]] = vector.broadcast %[[ARG1]] : index to vector<32xindex>
// CHECK:   %[[ADDI2:.*]] = arith.addi %[[ADDI]], %[[BCAST2]] : vector<32xindex>
// CHECK:   %[[CAST:.*]] = arith.index_cast %[[ADDI2]] : vector<32xindex> to vector<32xi32>
// CHECK:   vector.transfer_write %[[CAST]], %[[EMPTY]][%[[C0:.*]]] {in_bounds = [true]} : vector<32xi32>, tensor<32xi32>

transform.sequence failures(propagate) {
 ^bb1(%arg1: !pdl.operation):
   %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
   %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
   %2 = transform.structured.vectorize %1 { vectorize_nd_extract }
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func.func @test_vectorize_fill(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.fill ins(%arg0 : f32) outs(%A : memref<8x16xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func.func @test_vectorize_fill_scalar(%A : memref<f32>, %arg0 : f32) {
  // CHECK-SAME: (%[[M:.*]]: memref<f32>, %[[val:.*]]: f32)
  //      CHECK:   %[[VEC:.*]] = vector.broadcast %[[val]] : f32 to vector<f32>
  //      CHECK:   vector.transfer_write %[[VEC]], %[[M]][] : vector<f32>, memref<f32>
  linalg.fill ins(%arg0 : f32) outs(%A : memref<f32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_vectorize_copy
func.func @test_vectorize_copy(%A : memref<8x16xf32>, %B : memref<8x16xf32>) {
  //       CHECK: %[[V:.*]] = vector.transfer_read {{.*}} : memref<8x16xf32>, vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %A, %B :  memref<8x16xf32> to memref<8x16xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["memref.copy"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_vectorize_copy_scalar
func.func @test_vectorize_copy_scalar(%A : memref<f32>, %B : memref<f32>) {
  //  CHECK-SAME: (%[[A:.*]]: memref<f32>, %[[B:.*]]: memref<f32>)
  //       CHECK:   %[[V:.*]] = vector.transfer_read %[[A]][]{{.*}} : memref<f32>, vector<f32>
  //       CHECK:   %[[val:.*]] = vector.extractelement %[[V]][] : vector<f32>
  //       CHECK:   %[[VV:.*]] = vector.broadcast %[[val]] : f32 to vector<f32>
  //       CHECK:   vector.transfer_write %[[VV]], %[[B]][] : vector<f32>, memref<f32>
  memref.copy %A, %B :  memref<f32> to memref<f32>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["memref.copy"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_vectorize_copy_complex
// CHECK-NOT: vector<
func.func @test_vectorize_copy_complex(%A : memref<8x16xcomplex<f32>>, %B : memref<8x16xcomplex<f32>>) {
  memref.copy %A, %B :  memref<8x16xcomplex<f32>> to memref<8x16xcomplex<f32>>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["memref.copy"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_vectorize_trailing_index
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<1x2x4x8xindex>)
func.func @test_vectorize_trailing_index(%arg0: memref<1x2x4x8xindex>) {
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%arg0: memref<1x2x4x8xindex>) {
  ^bb0(%arg1: index):
  //       CHECK:   %[[BCST:.*]] = vector.broadcast %[[CST0]] : vector<8xindex> to vector<1x2x4x8xindex>
  //       CHECK:   vector.transfer_write %[[BCST]], %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {{.*}} : vector<1x2x4x8xindex>, memref<1x2x4x8xindex>
    %0 = linalg.index 3 : index
    linalg.yield %0 : index
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @test_vectorize_inner_index
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<1x2x4x8xindex>)
func.func @test_vectorize_inner_index(%arg0: memref<1x2x4x8xindex>) {
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<[0, 1]> : vector<2xindex>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%arg0: memref<1x2x4x8xindex>) {
  ^bb0(%arg1: index):
  //       CHECK:   %[[BCST:.*]] = vector.broadcast %[[CST0]] : vector<2xindex> to vector<1x8x4x2xindex>
  //       CHECK:   %[[TRAN:.*]] = vector.transpose %[[BCST]], [0, 3, 2, 1] : vector<1x8x4x2xindex> to vector<1x2x4x8xindex>
  //       CHECK:   vector.transfer_write %[[TRAN]], %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {{.*}} : vector<1x2x4x8xindex>, memref<1x2x4x8xindex>
    %0 = linalg.index 1 : index
    linalg.yield %0 : index
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// CHECK-LABEL: func @generic_vectorize
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<4x256xf32>, %[[ARG1:.*]]: memref<4x256xf32>,
  //  CHECK-SAME:  %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: f32)
func.func @generic_vectorize(%arg0: memref<4x256xf32>,
                        %arg1: memref<4x256xf32>,
                        %arg2: memref<256xf32>, %i: f32) {
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[CST1:.*]] = arith.constant dense<1.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  %c1_f32 = arith.constant 1.0 : f32
  linalg.generic {
    args_in = 0 : i64,
    args_out = 10 : i64,
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg1, %arg2: memref<4x256xf32>, memref<256xf32>)
  outs(
    %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 :
    memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>,
    memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>,
    memref<4x256xf32>, memref<4x256xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32,
  //       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : memref<256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //       CHECK:   %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : vector<4x256xf32>
    %6 = arith.addf %arg4, %arg6 : f32
  //       CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[V2]], %[[V1]] : vector<4x256xf32>
    %7 = arith.cmpf ogt, %arg3, %arg6 : f32
  //       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
    %8 = arith.constant 2.0 : f32
  //       CHECK:   %[[DIV:.*]] = arith.divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
    %9 = arith.divf %arg5, %i : f32
  //       CHECK:   %[[EXP:.*]] = math.exp2 %[[V3]] : vector<4x256xf32>
    %10 = math.exp2 %arg5 : f32
  //       CHECK:   %[[MUL:.*]] = arith.mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
    %11 = arith.mulf %arg5, %8 : f32
  //       CHECK:   %[[RSQRT:.*]] = math.rsqrt %[[V3]] : vector<4x256xf32>
    %12 = math.rsqrt %arg5 : f32
  //       CHECK:   %[[SEL:.*]] = arith.select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
    %13 = arith.select %7, %arg5, %arg6 : f32
  //       CHECK:   %[[SUB:.*]] = arith.subf %[[V3]], %[[V0]] : vector<4x256xf32>
    %14 = arith.subf %arg5, %arg4 : f32
  //       CHECK:   %[[TAN:.*]] = math.tanh %[[V3]] : vector<4x256xf32>
    %15 = math.tanh %arg5 : f32
  //       CHECK:   vector.transfer_write %[[ADD]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[CST0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[CST1]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[DIV]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[EXP]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[MUL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[RSQRT]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[SEL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[SUB]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[TAN]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_transfer_permutation_map_lowering_patterns }
}

// -----

// CHECK-LABEL: func @generic_vectorize_tensor
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<4x256xf32>, %[[ARG1:.*]]: tensor<4x256xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: tensor<256xf32>, %[[ARG3:.*]]: f32)
func.func @generic_vectorize_tensor(%arg0: tensor<4x256xf32>,
  %arg1: tensor<4x256xf32>, %arg2: tensor<256xf32>,
  %i: f32) -> (tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>) {
  %c1_f32 = arith.constant 1.0 : f32
  %r:10 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg1, %arg2: tensor<4x256xf32>, tensor<256xf32>)
  outs(
    %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 :
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32,
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[CST1:.*]] = arith.constant dense<1.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  //       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : tensor<256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : vector<4x256xf32>
    %6 = arith.addf %arg4, %arg6 : f32
  //       CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[V2]], %[[V1]] : vector<4x256xf32>
    %7 = arith.cmpf ogt, %arg3, %arg6 : f32
  //       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
    %8 = arith.constant 2.0 : f32
  //       CHECK:   %[[DIV:.*]] = arith.divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
    %9 = arith.divf %arg5, %i : f32
  //       CHECK:   %[[EXP:.*]] = math.exp2 %[[V3]] : vector<4x256xf32>
    %10 = math.exp2 %arg5 : f32
  //       CHECK:   %[[MUL:.*]] = arith.mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
    %11 = arith.mulf %arg5, %8 : f32
  //       CHECK:   %[[RSQRT:.*]] = math.rsqrt %[[V3]] : vector<4x256xf32>
    %12 = math.rsqrt %arg5 : f32
  //       CHECK:   %[[SEL:.*]] = arith.select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
    %13 = arith.select %7, %arg5, %arg6 : f32
  //       CHECK:   %[[SUB:.*]] = arith.subf %[[V3]], %[[V0]] : vector<4x256xf32>
    %14 = arith.subf %arg5, %arg4 : f32
  //       CHECK:   %[[TAN:.*]] = math.tanh %[[V3]] : vector<4x256xf32>
    %15 = math.tanh %arg5 : f32
  //       CHECK:   %[[R0:.*]] = vector.transfer_write %[[ADD]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R1:.*]] = vector.transfer_write %[[CST0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R2:.*]] = vector.transfer_write %[[CST1]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R3:.*]] = vector.transfer_write %[[DIV]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R4:.*]] = vector.transfer_write %[[EXP]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R5:.*]] = vector.transfer_write %[[MUL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R6:.*]] = vector.transfer_write %[[RSQRT]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R7:.*]] = vector.transfer_write %[[SEL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R8:.*]] = vector.transfer_write %[[SUB]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R9:.*]] = vector.transfer_write %[[TAN]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  } -> (tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>)
  //       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]], %[[R8]], %[[R9]] : tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
  return %r#0, %r#1, %r#2, %r#3, %r#4, %r#5, %r#6, %r#7, %r#8, %r#9:
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1 { disable_transfer_permutation_map_lowering_patterns }
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, 0, 0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0, 0, 0, 0)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1, 0, d0, 0)>
//     CHECK: func @generic_vectorize_broadcast_transpose
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[CF:.*]] = arith.constant 0.000000e+00 : f32
//     CHECK:   %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP0]]} : memref<4x4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP1]]} : memref<4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V2:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP2]]} : memref<4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V3:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP3]]} : memref<4x4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[SUB:.*]] = arith.subf %[[V0]], %[[V1]] : vector<4x4x4x4xf32>
//     CHECK:   %[[ADD0:.*]] = arith.addf %[[V2]], %[[SUB]] : vector<4x4x4x4xf32>
//     CHECK:   %[[ADD1:.*]] = arith.addf %[[V3]], %[[ADD0]] : vector<4x4x4x4xf32>
//     CHECK: vector.transfer_write %[[ADD1]], {{.*}} : vector<4x4x4x4xf32>, memref<4x4x4x4xf32>
func.func @generic_vectorize_broadcast_transpose(
  %A: memref<4xf32>, %B: memref<4x4xf32>, %C: memref<4x4x4x4xf32>) {
  linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
                   affine_map<(d0, d1, d2, d3) -> (d0)>,
                   affine_map<(d0, d1, d2, d3) -> (d2)>,
                   affine_map<(d0, d1, d2, d3) -> (d2, d0)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%B, %A, %A, %B: memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4x4xf32>)
  outs(%C : memref<4x4x4x4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %s = arith.subf %arg0, %arg1 : f32
    %a = arith.addf %arg2, %s : f32
    %b = arith.addf %arg3, %a : f32
    linalg.yield %b : f32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_transfer_permutation_map_lowering_patterns }
}

// -----

// Test different input maps.
#matmul_trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0, 0, 0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (0, d1, 0, d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3, d0)>
//       CHECK: func @vectorization_transpose
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP0]]} : memref<14x7xf32>, vector<7x14x8x16xf32>
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP1]]} : memref<16x14xf32>, vector<7x14x8x16xf32>
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP2]]} : memref<16x14x7x8xf32>, vector<7x14x8x16xf32>
//       CHECK: arith.addf {{.*}} : vector<7x14x8x16xf32>
//       CHECK: arith.addf {{.*}} : vector<7x14x8x16xf32>
//       CHECK: vector.transfer_write {{.*}} : vector<7x14x8x16xf32>, memref<7x14x8x16xf32>
func.func @vectorization_transpose(%A: memref<14x7xf32>, %B: memref<16x14xf32>,
                         %C: memref<16x14x7x8xf32>, %D: memref<7x14x8x16xf32>) {
  linalg.generic #matmul_trait
    ins(%A, %B, %C : memref<14x7xf32>, memref<16x14xf32>, memref<16x14x7x8xf32>)
   outs(%D : memref<7x14x8x16xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32, %d: f32) :
      %e = arith.addf %a, %b: f32
      %f = arith.addf %e, %c: f32
      linalg.yield %f : f32
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_transfer_permutation_map_lowering_patterns }
}

// -----

// CHECK-LABEL: func @matmul_tensors
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32>, %[[ARG1:.*]]: tensor<4x12xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: tensor<8x12xf32>) -> tensor<8x12xf32>
func.func @matmul_tensors(
  %arg0: tensor<8x4xf32>, %arg1: tensor<4x12xf32>, %arg2: tensor<8x12xf32>)
    -> tensor<8x12xf32> {
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  //   CHECK-DAG:   %[[V0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x4xf32>, vector<8x12x4xf32>
  //   CHECK-DAG:   %[[V1:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x12xf32>, vector<8x12x4xf32>
  //   CHECK-DAG:   %[[V2:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x12xf32>, vector<8x12xf32>
  //
  // linalg matmul lowers gets expanded to a 3D reduction, canonicalization later
  // convert it to a 2D contract.
  //       CHECK:   %[[MUL:.*]] = arith.mulf %[[V0]], %[[V1]] : vector<8x12x4xf32>
  //       CHECK:   %[[R:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[V2]] [2] : vector<8x12x4xf32> to vector<8x12xf32>
  //       CHECK:   %[[W:.*]] = vector.transfer_write %[[R]], %[[ARG2]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<8x12xf32>, tensor<8x12xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<8x4xf32>, tensor<4x12xf32>)
                     outs(%arg2: tensor<8x12xf32>)
    -> tensor<8x12xf32>
  //       CHECK:   return %[[W]] : tensor<8x12xf32>
  return %0 : tensor<8x12xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

// CHECK-LABEL: func @pad_static(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x?x2xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<2x3x4xf32>
//   CHECK-DAG:   %[[VEC:.*]] = vector.broadcast %[[PAD]] : f32 to vector<2x3x4xf32>
//       CHECK:   %[[FILL:.*]] = vector.transfer_write %[[VEC]], %[[INIT]]{{.*}} : vector<2x3x4xf32>, tensor<2x3x4xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] {in_bounds = [true, false, true]} : tensor<2x?x2xf32>, vector<2x3x2xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C0]], %[[C0]], %[[C2]]] {in_bounds = [true, true, true]} : vector<2x3x2xf32>, tensor<2x3x4xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_static(%arg0: tensor<2x?x2xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<2x?x2xf32> to tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1 { vectorize_padding }
}

// -----

// CHECK-LABEL: func @pad_static_source(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x5x2xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[INIT:.*]] = tensor.empty() : tensor<2x6x4xf32>
//       CHECK:   %[[VEC:.*]] =  vector.broadcast %[[PAD]] : f32 to vector<2x6x4xf32>
//       CHECK:   %[[FILL:.*]] = vector.transfer_write %[[VEC]], %[[INIT]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<2x6x4xf32>, tensor<2x6x4xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : tensor<2x5x2xf32>, vector<2x5x2xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C0]], %[[C0]], %[[C2]]] {in_bounds = [true, true, true]} : vector<2x5x2xf32>, tensor<2x6x4xf32>
//       CHECK:   return %[[WRITE]]
func.func @pad_static_source(%arg0: tensor<2x5x2xf32>, %pad_value: f32) -> tensor<2x6x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<2x5x2xf32> to tensor<2x6x4xf32>
  return %0 : tensor<2x6x4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { vectorize_padding }
}


// -----

// CHECK-LABEL: func @pad_static_dynamic(
//  CHECK-SAME:                          %[[SRC:.*]]: tensor<1x2x2x?xf32>, %[[LOW:.*]]: index, %[[HIGH:.*]]: index
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
//       CHECK:   %[[V0:.*]] = arith.addi %[[LOW]], %[[C2]] : index
//       CHECK:   %[[V1:.*]] = arith.addi %[[V0]], %[[C3]] : index
//       CHECK:   %[[V2:.*]] = arith.addi %[[HIGH]], %[[C5]] : index
//       CHECK:   %[[DIM3:.*]] = tensor.dim %[[SRC]], %[[C3]] : tensor<1x2x2x?xf32>
//       CHECK:   %[[V4:.*]] = arith.addi %[[DIM3]], %[[C3]] : index
//       CHECK:   %[[V5:.*]] = arith.addi %[[V4]], %[[C2]] : index
//       CHECK:   %[[INIT:.*]] = tensor.empty(%[[V1]], %[[V2]], %[[V5]]) : tensor<6x?x?x?xf32>
//       CHECK:   %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[INIT]] : tensor<6x?x?x?xf32>) -> tensor<6x?x?x?xf32>
//       CHECK:   %[[SRCDIM:.*]] = tensor.dim %[[SRC]], %[[C3]] : tensor<1x2x2x?xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.insert_slice %[[SRC]] into %[[FILL]][2, %[[LOW]], 3, 3] [1, 2, 2, %[[SRCDIM]]] [1, 1, 1, 1] : tensor<1x2x2x?xf32> into tensor<6x?x?x?xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_static_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  %0 = tensor.pad %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { vectorize_padding }
}

// -----

// CHECK-LABEL: func @pad_static_complex(
//   CHECK-NOT:   vector<
func.func @pad_static_complex(%arg0: tensor<2x5x2xcomplex<f32>>, %pad_value: complex<f32>) -> tensor<2x6x4xcomplex<f32>> {
  %0 = tensor.pad %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : complex<f32>
    } : tensor<2x5x2xcomplex<f32>> to tensor<2x6x4xcomplex<f32>>
  return %0 : tensor<2x6x4xcomplex<f32>>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { vectorize_padding }
}

// -----

// CHECK-LABEL: func @pad_and_transfer_read
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5.0
//       CHECK:   %[[RESULT:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[C5]] : tensor<5x6xf32>, vector<7x9xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_and_transfer_read(%arg0: tensor<5x6xf32>) -> vector<7x9xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %c6 = arith.constant 6.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %c6
      : tensor<10x13xf32>, vector<7x9xf32>
  return %1 : vector<7x9xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1 { vectorize_padding }
}

// -----

func.func private @make_vector() -> vector<7x9xf32>

// CHECK-LABEL: func @pad_and_transfer_write_static
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[VEC0:.*]] = call @make_vector() : () -> vector<7x9xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[VEC0]], %[[ARG0]][%[[C0]], %[[C0]]] : vector<7x9xf32>, tensor<5x6xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_and_transfer_write_static(
    %arg0: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = call @make_vector() : () -> vector<7x9xf32>
  %2 = vector.transfer_write %1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<10x13xf32>
  %3 = tensor.extract_slice %2[0, 0] [5, 6] [1, 1] : tensor<10x13xf32> to tensor<5x6xf32>
  return %3 : tensor<5x6xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4  { vectorize_padding }
}


// -----

func.func private @make_vector() -> vector<7x9xf32>

// CHECK-LABEL: func @pad_and_transfer_write_dynamic_static
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[SIZE:.*]]: index, %[[PADDING:.*]]: index
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[SUB:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[SIZE]], 6] [1, 1] : tensor<?x?xf32> to tensor<?x6xf32>
//       CHECK:   %[[VEC0:.*]] = call @make_vector() : () -> vector<7x9xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[VEC0]], %[[SUB]][%[[C0]], %[[C0]]] : vector<7x9xf32>, tensor<?x6xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_and_transfer_write_dynamic_static(
    %arg0: tensor<?x?xf32>, %size: index, %padding: index) -> tensor<?x6xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %s = tensor.extract_slice %arg0[0, 0] [%size, 6] [1, 1]
      : tensor<?x?xf32> to tensor<?x6xf32>
  %0 = tensor.pad %s low[0, 0] high[%padding, 7] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<?x6xf32> to tensor<?x13xf32>
  %1 = call @make_vector() : () -> vector<7x9xf32>
  %2 = vector.transfer_write %1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<?x13xf32>
  %3 = tensor.extract_slice %2[0, 0] [%size, 6] [1, 1] : tensor<?x13xf32> to tensor<?x6xf32>
  return %3 : tensor<?x6xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4 { vectorize_padding }
}


// -----

func.func private @make_vector() -> tensor<12x13xf32>

// CHECK-LABEL: func @pad_and_insert_slice_source
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5.0
//       CHECK:   %[[VEC0:.*]] = call @make_vector() : () -> tensor<12x13xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[C5]] : tensor<5x6xf32>, vector<7x9xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[VEC0]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<7x9xf32>, tensor<12x13xf32>
//       CHECK:   return %[[WRITE]]
func.func @pad_and_insert_slice_source(
    %arg0: tensor<5x6xf32>) -> tensor<12x13xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[2, 3] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<7x9xf32>
  %1 = call @make_vector() : () -> tensor<12x13xf32>
  %r = tensor.insert_slice %0 into %1[0, 0][7, 9][1, 1] : tensor<7x9xf32> into tensor<12x13xf32>
  return %r : tensor<12x13xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4  { vectorize_padding }
}


// -----

func.func private @make_vector() -> tensor<12x13xf32>

// CHECK-LABEL: func @pad_and_insert_slice_dest
// Check the insert slice is not rewritten if the padded result is used by the destination operand.
//       CHECK:   %[[T1:.*]] = call @make_vector() : () -> tensor<12x13xf32>
//       CHECK:   = tensor.insert_slice %[[T1]] into
func.func @pad_and_insert_slice_dest(
    %arg0: tensor<1x5x6xf32>) -> tensor<1x12x13xf32> {
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0, 0] high[0, 7, 7] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      tensor.yield %c5 : f32
  } : tensor<1x5x6xf32> to tensor<1x12x13xf32>
  %1 = call @make_vector() : () -> tensor<12x13xf32>
  %r = tensor.insert_slice %1 into %0[0, 0, 0][1, 12, 13][1, 1, 1] : tensor<12x13xf32> into tensor<1x12x13xf32>
  return %r : tensor<1x12x13xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-LABEL: func @pad_tensor_non_const_pad_value
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//       CHECK:   %[[FILL:.*]] = tensor.generate
//       CHECK:     %[[RES:.*]] = arith.mulf
//       CHECK:     tensor.yield %[[RES]] : f32
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : tensor<5x6xf32>, vector<5x6xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C3]], %[[C4]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<12x13xf32>
//       CHECK:   return %[[WRITE]]
func.func @pad_tensor_non_const_pad_value(%arg0: tensor<5x6xf32>) -> tensor<12x13xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[3, 4] high[4, 3] {
    ^bb0(%arg1: index, %arg2: index):
      %i1 = arith.index_cast %arg1 : index to i32
      %i2 = arith.index_cast %arg2 : index to i32
      %f1 = arith.sitofp %i1 : i32 to f32
      %f2 = arith.sitofp %i2 : i32 to f32
      %m = arith.mulf %f1, %f2 : f32
      tensor.yield %m : f32
  } : tensor<5x6xf32> to tensor<12x13xf32>
  return %0 : tensor<12x13xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4  { vectorize_padding }
}

// -----

// CHECK-LABEL: func @sum_exp
func.func @sum_exp(%input: tensor<4x16x8xf32>, %output: tensor<4x16xf32>)
  -> tensor<4x16xf32>
{
  // CHECK: vector.transfer_read {{.*}} : tensor<4x16x8xf32>, vector<4x16x8xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<4x16xf32>, vector<4x16xf32>
  // CHECK: math.exp {{.*}} : vector<4x16x8xf32>
  // CHECK: vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<4x16x8xf32> to vector<4x16xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4x16xf32>, tensor<4x16xf32>
  // CHECK: return {{.*}} : tensor<4x16xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input : tensor<4x16x8xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %1 = math.exp %arg0 : f32
      %2 = arith.addf %1, %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-DAG: #[[$M1:.*]] =  affine_map<(d0, d1) -> (d1, d0, 0, 0)>
// CHECK-DAG: #[[$M2:.*]] =  affine_map<(d0, d1) -> (0, 0, d1, d0)>
// CHECK-DAG: #[[$M3:.*]] =  affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @sum_exp_2
func.func @sum_exp_2(%input: tensor<3x2xf32>, %input_2: tensor<5x4xf32>, %output: tensor<5x2xf32>)
  -> tensor<5x2xf32>
{
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M1]]} : tensor<3x2xf32>, vector<2x3x4x5xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M2]]} : tensor<5x4xf32>, vector<2x3x4x5xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #[[$M3]]} : tensor<5x2xf32>, vector<2x5xf32>
  // CHECK: math.exp {{.*}} : vector<2x3x4x5xf32>
  // CHECK: math.exp {{.*}} : vector<2x3x4x5xf32>
  // CHECK: addf {{.*}} : vector<2x3x4x5xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}}, %{{.*}}  [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true, true], permutation_map = #[[$M3]]} : vector<2x5xf32>, tensor<5x2xf32>
  // CHECK: return {{.*}} : tensor<5x2xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d0)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0)>
      ],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]
    } ins(%input, %input_2 : tensor<3x2xf32>, tensor<5x4xf32>) outs(%output : tensor<5x2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %1 = math.exp %arg0 : f32
      %2 = math.exp %arg1 : f32
      %3 = arith.addf %1, %2 : f32
      %4 = arith.addf %3, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

// CHECK-LABEL:   func @red_max_2d(
func.func @red_max_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMINF:.+]] = arith.constant dense<-3.402820e+38> : vector<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.multi_reduction <maxf>, {{.*}}, %[[CMINF]] [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %ident = arith.constant -3.40282e+38 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%ident : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %max = arith.maxf %in0, %out0 : f32
    linalg.yield %max : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4 { vectorize_padding }
}

// -----

// CHECK-LABEL:   func @red_min_2d(
func.func @red_min_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMAXF:.+]] = arith.constant dense<3.402820e+38> : vector<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.multi_reduction <minf>, {{.*}}, %[[CMAXF]] [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %maxf32 = arith.constant 3.40282e+38 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%maxf32 : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %min = arith.minf %out0, %in0 : f32
    linalg.yield %min : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-LABEL:   func @red_mul_2d(
func.func @red_mul_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.multi_reduction <mul>, {{.*}}, {{.*}} [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %ident = arith.constant 1.0 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%ident : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %mul = arith.mulf %in0, %out0 : f32
    linalg.yield %mul : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-LABEL:   func @red_or_2d(
func.func @red_or_2d(%arg0: tensor<4x4xi1>) -> tensor<4xi1> {
  // CHECK: tensor.empty() : tensor<4xi1>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xi1>, vector<4x4xi1>
  // CHECK: vector.multi_reduction <or>, {{.*}}, {{.*}} [1] : vector<4x4xi1> to vector<4xi1>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi1>, tensor<4xi1>
  %ident = arith.constant false
  %init = tensor.empty() : tensor<4xi1>
  %fill = linalg.fill ins(%ident : i1) outs(%init : tensor<4xi1>) -> tensor<4xi1>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xi1>) outs(%fill : tensor<4xi1>) {
  ^bb0(%in0: i1, %out0: i1):
    %or = arith.ori %in0, %out0 : i1
    linalg.yield %or : i1
  } -> tensor<4xi1>
  return %red : tensor<4xi1>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-LABEL:   func @red_and_2d(
func.func @red_and_2d(%arg0: tensor<4x4xi1>) -> tensor<4xi1> {
  // CHECK: tensor.empty() : tensor<4xi1>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xi1>, vector<4x4xi1>
  // CHECK: vector.multi_reduction <and>, {{.*}}, {{.*}} [1] : vector<4x4xi1> to vector<4xi1>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi1>, tensor<4xi1>
  %ident = arith.constant true
  %init = tensor.empty() : tensor<4xi1>
  %fill = linalg.fill ins(%ident : i1) outs(%init : tensor<4xi1>) -> tensor<4xi1>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xi1>) outs(%fill : tensor<4xi1>) {
  ^bb0(%in0: i1, %out0: i1):
    %and = arith.andi %in0, %out0 : i1
    linalg.yield %and : i1
  } -> tensor<4xi1>
  return %red : tensor<4xi1>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-LABEL:   func @red_xor_2d(
func.func @red_xor_2d(%arg0: tensor<4x4xi1>) -> tensor<4xi1> {
  // CHECK: tensor.empty() : tensor<4xi1>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xi1>, vector<4x4xi1>
  // CHECK: vector.multi_reduction <xor>, {{.*}}, {{.*}} [1] : vector<4x4xi1> to vector<4xi1>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi1>, tensor<4xi1>
  %ident = arith.constant false
  %init = tensor.empty() : tensor<4xi1>
  %fill = linalg.fill ins(%ident : i1) outs(%init : tensor<4xi1>) -> tensor<4xi1>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xi1>) outs(%fill : tensor<4xi1>) {
  ^bb0(%in0: i1, %out0: i1):
    %xor = arith.xori %in0, %out0 : i1
    linalg.yield %xor : i1
  } -> tensor<4xi1>
  return %red : tensor<4xi1>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-DAG: #[[$M5:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL:   func @explicit_broadcast(
func.func @explicit_broadcast(%arg0: tensor<4x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x4xf32> {
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #[[$M5]]} : tensor<4x1xf32>, vector<4x4xf32>
  // CHECK: subf {{.*}} : vector<4x4xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, 0)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]}
   ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x1xf32>)
   outs(%fill : tensor<4x4xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %40 = arith.subf %arg7, %arg8 : f32
      linalg.yield %40 : f32
    } -> tensor<4x4xf32>
  return %red : tensor<4x4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

// CHECK-DAG: #[[$M6:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL:   func @fused_broadcast_red_2d
func.func @fused_broadcast_red_2d(%arg0: tensor<4x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4xf32> {
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #[[$M6]]} : tensor<4x1xf32>, vector<4x4xf32>
  // CHECK: subf {{.*}} : vector<4x4xf32>
  // CHECK: math.exp {{.*}} : vector<4x4xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}}, {{.*}} : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, 0)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x1xf32>)
   outs(%fill : tensor<4xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %40 = arith.subf %arg7, %arg8 : f32
      %41 = math.exp %40 : f32
      %42 = arith.addf %41, %arg9 : f32
      linalg.yield %42 : f32
    } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1

  %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %4 = get_closest_isolated_parent %3 : (!pdl.operation) -> !pdl.operation
  %5 = transform.structured.vectorize %4
}

// -----

//  CHECK-LABEL: func @reduce_1d(
//   CHECK-SAME:   %[[A:.*]]: tensor<32xf32>
func.func @reduce_1d(%arg0: tensor<32xf32>) -> tensor<f32> {
  //  CHECK-DAG: %[[vF0:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
  //  CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32

  //      CHECK: %[[init:.*]] = tensor.empty() : tensor<f32>
  %0 = tensor.empty() : tensor<f32>

  %1 = linalg.fill ins(%f0 : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  //      CHECK: %[[r:.*]] = vector.transfer_read %[[A]][%[[C0]]]
  // CHECK-SAME:   : tensor<32xf32>, vector<32xf32>
  //      CHECK: %[[f0:.*]] = vector.extractelement %[[vF0]][] : vector<f32>
  //      CHECK: %[[red:.*]] = vector.multi_reduction <add>, %[[r]], %[[f0]] [0]
  // CHECK-SAME:   : vector<32xf32> to f32
  //      CHECK: %[[red_v1:.*]] = vector.broadcast %[[red]] : f32 to vector<f32>
  //      CHECK: %[[res:.*]] = vector.transfer_write %[[red_v1]], %[[init]][]
  // CHECK-SAME:   : vector<f32>, tensor<f32>
  %2 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> ()>],
         iterator_types = ["reduction"]}
         ins(%arg0 : tensor<32xf32>)
         outs(%1 : tensor<f32>) {
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      linalg.yield %3 : f32
    } -> tensor<f32>

  return %2 : tensor<f32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}


// -----

// This test checks that vectorization does not occur when an input indexing map
// is not a projected permutation. In the future, this can be converted to a
// positive test when support is added.

// CHECK-LABEL:   func @not_projected_permutation
func.func @not_projected_permutation(%arg0: tensor<8x8xf32>) -> tensor<6x6x3x3xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<6x6x3x3xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<6x6x3x3xf32>) -> tensor<6x6x3x3xf32>
  // CHECK: linalg.generic
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
   iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
   ins(%arg0 : tensor<8x8xf32>)
   outs(%fill : tensor<6x6x3x3xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      linalg.yield %arg7 : f32
    } -> tensor<6x6x3x3xf32>
  return %result : tensor<6x6x3x3xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

// Check vectorization can handle cases where outputs are a mix of reduced and non-reduced values.
func.func @mixed_parallel_reduced_results(%arg0 : tensor<2x4x8xf32>,
    %arg1 : tensor<2x4xf32>, %arg2 : tensor<2x4x8xf32>, %arg3 : tensor<2x4xf32>) ->
    (tensor<2x4x8xf32>, tensor<2x4xf32>) {
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<2x4x8xf32>, tensor<2x4xf32>)
      outs(%arg2, %arg3 : tensor<2x4x8xf32>, tensor<2x4xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32, %b3 : f32):
      %1 = arith.mulf %b0, %b1 : f32
      %2 = arith.addf %1, %b3 : f32
      linalg.yield %1, %2 : f32, f32
  } -> (tensor<2x4x8xf32>, tensor<2x4xf32>)
  return %0#0, %0#1 : tensor<2x4x8xf32>, tensor<2x4xf32>
}
// CHECK-LABEL: func @mixed_parallel_reduced_results(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x4x8xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x4xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<2x4x8xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<2x4xf32>
//   CHECK-DAG:   %[[V0:.+]] = vector.transfer_read %[[ARG0]]
//   CHECK-DAG:   %[[V1:.+]] = vector.transfer_read %[[ARG1]]
//   CHECK-DAG:   %[[V2:.+]] = vector.transfer_read %[[ARG3]]
//   CHECK-DAG:   %[[MUL:.+]] = arith.mulf %[[V0]], %[[V1]]
//   CHECK-DAG:   %[[ADD:.+]] = vector.multi_reduction <add>, %[[MUL]], %[[V2]]
//   CHECK-DAG:   vector.transfer_write %[[MUL]], %[[ARG2]]
//   CHECK-DAG:   vector.transfer_write %[[ADD]], %[[ARG3]]

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @vectorize_1d_tensor_extract(%arg0: tensor<3xf32>, %arg1: tensor<4x3xi32>, %arg2: tensor<4x7x3x2xf32>) -> tensor<4x7x3x2xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<4x3xi32>) outs(%arg2 : tensor<4x7x3x2xf32>) {
  ^bb0(%arg3: i32, %arg4: f32):
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = tensor.extract %arg0[%2] : tensor<3xf32>
    linalg.yield %3 : f32
  } -> tensor<4x7x3x2xf32>
  return %1 : tensor<4x7x3x2xf32>
}
// CHECK-LABEL: func.func @vectorize_1d_tensor_extract
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3xf32>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<4x3xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<4x7x3x2xi1>
// CHECK: %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<4x7x3x2xf32>
// CHECK: %[[V0:.*]] = vector.transfer_read %[[ARG1]]
// CHECK: %[[CAST:.*]] = arith.index_cast %[[V0]]
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[CAST]]
// CHECK: %[[INDICES:.*]] = vector.transpose %[[BROADCAST]]
// CHECK: %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]]] [%[[INDICES]]], %[[MASK]], %[[PASSTHRU]]
// CHECK: vector.transfer_write %[[GATHER]]

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_nd_tensor_extract_constant_idx(%arg0: tensor<3x3xf32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  %2 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %3 = linalg.index 2 : index
    %7 = tensor.extract %arg0[%c0, %c1] : tensor<3x3xf32>
    linalg.yield %7 : f32
  } -> tensor<1x1x3xf32>
  return %2 : tensor<1x1x3xf32>
}

// CHECK-LABEL: func.func @vectorize_nd_tensor_extract_constant_idx
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x1x3xf32>
// CHECK:    %[[MASK:.*]] = arith.constant dense<true> : vector<1x1x3xi1>
// CHECK:    %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x3xf32>
// CHECK:    %[[C0:.*]] = arith.constant 0 : index
// Magic "5" below comes from (1 * 3 + 2) (1: index into dim 1, 2: index into dim 2)
// CHECK:    %[[IDX:.*]] = arith.constant dense<5> : vector<1x1x3xindex>
// CHECK:    %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]], %[[C0]]] [%[[IDX]]], %[[MASK]], %[[PASSTHRU]] : tensor<3x3xf32>, vector<1x1x3xindex>, vector<1x1x3xi1>, vector<1x1x3xf32> into vector<1x1x3xf32>
// CHECK:    vector.transfer_write %[[GATHER]]
// CHECK:  }

transform.sequence failures(propagate) {
 ^bb1(%arg1: !pdl.operation):
   %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
   %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
   %2 = transform.structured.vectorize %1 { vectorize_nd_extract }
 }

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_nd_tensor_extract_idx_from_iteration_index(%arg0: tensor<3x3x3xf32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = linalg.index 2 : index
    %5 = tensor.extract %arg0[%2, %3, %4] : tensor<3x3x3xf32>
    linalg.yield %5 : f32
  } -> tensor<1x1x3xf32>
  return %1 : tensor<1x1x3xf32>
}

// CHECK-LABEL: func.func @vectorize_nd_tensor_extract_idx_from_iteration_index
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x3x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x1x3xf32>
// CHECK:   %[[INDICES:.*]] = arith.constant dense<[0, 1, 2]> : vector<3xindex>
// CHECK:   %[[MASK:.*]] = arith.constant dense<true> : vector<1x1x3xi1>
// CHECK:   %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x3xf32>
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[B:.*]] = vector.broadcast %[[INDICES]] : vector<3xindex> to vector<1x1x3xindex>
// CHECK:   %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] [%[[B]]], %[[MASK]], %[[PASSTHRU]] : tensor<3x3x3xf32>, vector<1x1x3xindex>, vector<1x1x3xi1>, vector<1x1x3xf32> into vector<1x1x3xf32>
// CHECK:   vector.transfer_write %[[GATHER]]

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1 { vectorize_nd_extract }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @vectorize_nd_tensor_extract_index_from_tensor(%arg0: tensor<3x3xf32>, %arg1: tensor<4x3xi32>, %arg2: tensor<4x3xi32>, %arg3: tensor<4x7x2xf32>, %arg4: tensor<4x7x3x2xf32>) -> tensor<4x7x3x2xf32> {
  %2 = linalg.generic {
    indexing_maps = [#map0, #map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1, %arg2, %arg3 : tensor<4x3xi32>, tensor<4x3xi32>, tensor<4x7x2xf32>) outs(%arg4 : tensor<4x7x3x2xf32>) {
  ^bb0(%arg5: i32, %arg6: i32, %arg7: f32, %arg8: f32):
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.index_cast %arg6 : i32 to index
    %7 = tensor.extract %arg0[%3, %4] : tensor<3x3xf32>
    linalg.yield %7 : f32
  } -> tensor<4x7x3x2xf32>
  return %2 : tensor<4x7x3x2xf32>
}
// CHECK-LABEL: func.func @vectorize_nd_tensor_extract_index_from_tensor
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x3xf32>
// CHECK-SAME: %[[ARG1:arg1]]: tensor<4x3xi32>
// CHECK-SAME: %[[ARG2:arg2]]: tensor<4x3xi32>
// CHECK-SAME: %[[ARG3:.*]]: tensor<4x7x2xf32>
// CHECK-SAME: %[[ARG4:.*]]: tensor<4x7x3x2xf32>
// CHECK:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:    %[[C0_i32:.*]] = arith.constant 0 : i32
// CHECK:    %[[CST:.*]] = arith.constant dense<3> : vector<7x2x4x3xindex>
// CHECK:    %[[CST_1:.*]] = arith.constant dense<true> : vector<4x7x3x2xi1>
// CHECK:    %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<4x7x3x2xf32>
// CHECK:    %[[V0:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], %[[C0_i32]] {in_bounds = [true, true]} : tensor<4x3xi32>, vector<4x3xi32>
// CHECK:    %[[V1:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], %[[C0_i32]] {in_bounds = [true, true]} : tensor<4x3xi32>, vector<4x3xi32>
// CHECK:    %[[CAST:.*]] = arith.index_cast %[[V0]] : vector<4x3xi32> to vector<4x3xindex>
// CHECK:    %[[B1:.*]] = vector.broadcast %[[CAST]] : vector<4x3xindex> to vector<7x2x4x3xindex>
// CHECK:    %[[CAST_1:.*]] = arith.index_cast %[[V1]] : vector<4x3xi32> to vector<4x3xindex>
// CHECK:    %[[B2:.*]] = vector.broadcast %[[CAST_1]] : vector<4x3xindex> to vector<7x2x4x3xindex>
// CHECK:    %[[MULI:.*]] = arith.muli %[[B1]], %[[CST]] : vector<7x2x4x3xindex>
// CHECK:    %[[ADDI:.*]] = arith.addi %[[B2]], %[[MULI]] : vector<7x2x4x3xindex>
// CHECK:    %[[T:.*]] = vector.transpose %[[ADDI]], [2, 0, 3, 1] : vector<7x2x4x3xindex> to vector<4x7x3x2xindex>
// CHECK:    %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]], %[[C0]]] [%[[T]]], %[[CST_1]], %[[PASSTHRU]] : tensor<3x3xf32>, vector<4x7x3x2xindex>, vector<4x7x3x2xi1>, vector<4x7x3x2xf32> into vector<4x7x3x2xf32>
// CHECK:    vector.transfer_write %[[GATHER]], %[[ARG4]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true, true]} : vector<4x7x3x2xf32>, tensor<4x7x3x2xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1 { vectorize_nd_extract }
}

// -----

func.func @vectorize_map(%arg0: memref<64xf32>,
    %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
  linalg.map ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>)
             outs(%arg2 : memref<64xf32>)
    (%in: f32, %in_0: f32) {
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
  return
}
// CHECK-LABEL: func @vectorize_map
// CHECK:         %[[LHS:.*]] = vector.transfer_read
// CHECK-NEXT:    %[[RHS:.*]] = vector.transfer_read
// CHECK-NEXT:    arith.addf %[[LHS]], %[[RHS]] : vector<64xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.map"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

func.func @vectorize_transpose(%arg0: memref<16x32x64xf32>,
                               %arg1: memref<32x64x16xf32>) {
  linalg.transpose ins(%arg0 : memref<16x32x64xf32>)
                   outs(%arg1 : memref<32x64x16xf32>) permutation = [1, 2, 0]
  return
}
// CHECK-LABEL: func @vectorize_transpose
// CHECK:         vector.transpose
// CHECK-SAME:      [1, 2, 0] : vector<16x32x64xf32> to vector<32x64x16xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

func.func @vectorize_reduce(%arg0: memref<16x32x64xf32>,
                  %arg1: memref<16x64xf32>) {
  linalg.reduce ins(%arg0 : memref<16x32x64xf32>)
                outs(%arg1 : memref<16x64xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  return
}
// CHECK-LABEL: func @vectorize_reduce
// CHECK:         vector.multi_reduction <add>
// CHECK-SAME:    : vector<16x32x64xf32> to vector<16x64xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// -----

func.func @vectorize_dynamic_identity(%arg0: tensor<?xf32>,
                                      %arg1: tensor<?xf32>,
                                      %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                   iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_identity
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_4]] : vector<4xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %0 vector_sizes [4]
}

// -----

func.func @vectorize_dynamic_1d_broadcast(%arg0: tensor<?xf32>,
                                          %arg1: tensor<?xf32>,
                                          %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                        iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_1d_broadcast
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %{{.*}} {permutation_map = #{{.*}}} : tensor<?xf32>, vector<4xf32>
// CHECK:           %[[VAL_9:.*]] = vector.create_mask %[[VAL_4]] : vector<4xi1>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_7]], %[[VAL_10]] : vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %{{.*}} { vector.transfer_write %[[VAL_13]], {{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %0 vector_sizes [4]
}

// -----

func.func @vectorize_dynamic_2d_transpose(%arg0: tensor<?x?xf32>,
                                          %arg1: tensor<?x?xf32>,
                                          %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_2d_transpose
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = tensor.dim %{{.*}}, %[[VAL_5]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = vector.create_mask %[[VAL_6]], %[[VAL_4]] : vector<8x4xi1>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<4x8xf32> } : vector<8x4xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_12:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]] : vector<4x8xi1>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_16:.*]] = arith.addf %[[VAL_10]], %[[VAL_13]] : vector<4x8xf32>
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_12]] { vector.transfer_write %[[VAL_16]], %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32> } : vector<4x8xi1> -> tensor<?x?xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %0 vector_sizes [4, 8]
}

// -----

func.func @vectorize_dynamic_generic_2d_broadcast(%arg0: tensor<?x?xf32>,
                                                  %arg1: tensor<?x?xf32>,
                                                  %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_generic_2d_broadcast
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = tensor.dim %{{.*}}, %[[VAL_5]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = vector.create_mask %[[VAL_6]] : vector<8xi1>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<4x8xf32> } : vector<8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_12:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]] : vector<4x8xi1>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_16:.*]] = arith.addf %[[VAL_10]], %[[VAL_13]] : vector<4x8xf32>
// CHECK:           %[[VAL_18:.*]] = vector.mask %[[VAL_12]] { vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32> } : vector<4x8xi1> -> tensor<?x?xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %0 vector_sizes [4, 8]
}

// -----

func.func @vectorize_dynamic_reduction(%arg0: tensor<?x?xf32>,
                                       %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0)>],
                        iterator_types = ["parallel", "reduction"] }
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %0 vector_sizes [4, 8]
}

// CHECK-LABEL:   @vectorize_dynamic_reduction(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]] : vector<4x8xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]]{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_11:.*]] = vector.create_mask %[[VAL_3]] : vector<4xi1>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_11]] { vector.transfer_read %[[VAL_1]]{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_8]] { vector.multi_reduction <add>, %[[VAL_9]], %[[VAL_12]] [1] : vector<4x8xf32> to vector<4xf32> } : vector<4x8xi1> -> vector<4xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_11]] { vector.transfer_write %[[VAL_13]], %[[VAL_1]]{{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
// CHECK:           return %[[VAL_15]] : tensor<?xf32>
// CHECK:         }

// -----

func.func @vectorize_dynamic_transpose_reduction(%arg0: tensor<?x?x?xf32>,
                                                 %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                         affine_map<(d0, d1, d2) -> (d2, d1)>],
                        iterator_types = ["reduction", "parallel", "parallel"] }
    ins(%arg0 : tensor<?x?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %0 vector_sizes [4, 8, 16]
}

// CHECK-LABEL:   @vectorize_dynamic_transpose_reduction(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]], %[[VAL_7]] : vector<4x8x16xi1>
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_10]] { vector.transfer_read %[[VAL_0]]{{.*}} {in_bounds = [true, true, true]} : tensor<?x?x?xf32>, vector<4x8x16xf32> } : vector<4x8x16xi1> -> vector<4x8x16xf32>
// CHECK:           %[[VAL_13:.*]] = vector.create_mask %[[VAL_7]], %[[VAL_5]] : vector<16x8xi1>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_13]] { vector.transfer_read %[[VAL_1]]{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_10]] { vector.multi_reduction <add>, %[[VAL_11]], %[[VAL_14]] [0] : vector<4x8x16xf32> to vector<8x16xf32> } : vector<4x8x16xi1> -> vector<8x16xf32>
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_13]] { vector.transfer_write %[[VAL_15]], %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<8x16xf32>, tensor<?x?xf32> } : vector<16x8xi1> -> tensor<?x?xf32>

// -----

// This is a regression test. This IR cannot be vectorized, but
// structured.vectorize should nevertheless succeed.

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   @not_vectorizable
func.func @not_vectorizable(%arg0: tensor<1x?xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<1x128xf32> {
  %0 = tensor.empty() : tensor<1x128xf32>
  %1 = scf.for %arg5 = %arg2 to %arg1 step %arg3 iter_args(%arg6 = %0) -> (tensor<1x128xf32>) {
    %extracted_slice = tensor.extract_slice %arg6[0, 0] [1, %arg1] [1, 1] : tensor<1x128xf32> to tensor<?xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, %arg3] [1, %arg2] [1, 1] : tensor<1x?xf32> to tensor<?xf32>
    %extracted_slice_1 = tensor.extract_slice %expanded[0, %arg3] [1, %arg2] [1, 1] : tensor<1x?xf32> to tensor<?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%extracted_slice_0 : tensor<?xf32>) outs(%extracted_slice_1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %2 into %expanded[0, %arg3] [1, %arg2] [1, 1] : tensor<?xf32> into tensor<1x?xf32>
    %collapsed = tensor.collapse_shape %inserted_slice [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    %inserted_slice_2 = tensor.insert_slice %collapsed into %arg6[0, 0] [1, %arg1] [1, 1] : tensor<?xf32> into tensor<1x128xf32>
    scf.yield %inserted_slice_2 : tensor<1x128xf32>
  }
  return %1 : tensor<1x128xf32>
}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.vectorize %0
}

// -----

// Regression test: %13 was incorrectly detected as a reduction and
// vectorization failed.

func.func @wrong_reduction_detection(%input: tensor<120x64xf32>) -> tensor<120x64xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst_6 = arith.constant 4.000000e+00 : f32
  %1 = scf.for %arg0 = %c0 to %c64 step %c4 iter_args(%arg1 = %input) -> (tensor<120x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%c0, %arg0] [1, 4] [1, 1] : tensor<120x64xf32> to tensor<1x4xf32>
    %10 = linalg.fill {__internal_linalg_transform__ = "1"} ins(%cst_6 : f32) outs(%extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%10 : tensor<1x4xf32>) {
    ^bb0(%out: f32):
      %12 = linalg.index 0 : index
      %13 = arith.addi %arg0, %12 : index
      %18 = arith.index_cast %13 : index to i32
      %20 = arith.uitofp %18 : i32 to f32
      %67 = arith.mulf %out, %20 : f32
      linalg.yield %67 : f32
    } -> tensor<1x4xf32>
    %inserted_slice = tensor.insert_slice %11 into %arg1[%c0, %arg0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<120x64xf32>
    scf.yield %inserted_slice : tensor<120x64xf32>
  }
  return %1 : tensor<120x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.vectorize %1
}

// CHECK-LABEL: @wrong_reduction_detection
// CHECK:         vector.broadcast
// CHECK:         vector.transfer_write
