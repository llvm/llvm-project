// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=category-to-named | \
// RUN:   FileCheck %s

func.func @elementwise_unary(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise kind = #linalg.elementwise_kind<exp>
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @elementwise_unary
// CHECK-SAME: %[[IN:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK-NOT: linalg.elementwise
// CHECK: linalg.exp
// CHECK-SAME: ins(%[[IN]] : tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @elementwise_binary(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise
      kind = #linalg.elementwise_kind<powf>
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @elementwise_binary
// CHECK-SAME: %[[LHS:.+]]: tensor<?x?xf32>, %[[RHS:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.elementwise
// CHECK: linalg.powf
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#elt_map_in = affine_map<(d0, d1) -> (d1, d0)>
#elt_map_out = affine_map<(d0, d1) -> (d0, d1)>

// This stays as `linalg.elementwise` because named elementwise ops do not
// support user-defined indexing maps.
func.func @elementwise_non_default_maps(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise
      kind = #linalg.elementwise_kind<exp>
      indexing_maps = [#elt_map_in, #elt_map_out]
      ins(%arg0 : tensor<?x?xf32>)
      outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[$ELT_MAP_IN:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$ELT_MAP_OUT:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @elementwise_non_default_maps
// CHECK: linalg.elementwise
// CHECK-SAME: indexing_maps = [#[[$ELT_MAP_IN]], #[[$ELT_MAP_OUT]]]
// CHECK-NOT: linalg.exp

// -----

#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @contract_to_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.contract indexing_maps = [#map_a, #map_b, #map_c]
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @contract_to_matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.contract
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#cast_map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#cast_map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#cast_map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @contract_to_matmul_unsigned_cast(%arg0: tensor<16x8xi16>,
    %arg1: tensor<8x32xi64>, %arg2: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.contract indexing_maps = [#cast_map_a, #cast_map_b, #cast_map_c]
      {cast = #linalg.type_fn<cast_unsigned>}
      ins(%arg0, %arg1 : tensor<16x8xi16>, tensor<8x32xi64>)
      outs(%arg2 : tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// CHECK-LABEL: @contract_to_matmul_unsigned_cast
// CHECK-SAME: %[[A:.+]]: tensor<16x8xi16>, %[[B:.+]]: tensor<8x32xi64>,
// CHECK-SAME: %[[OUT:.+]]: tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK-NOT: linalg.contract
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<16x8xi16>, tensor<8x32xi64>)
// CHECK-SAME: outs(%[[OUT]] : tensor<16x32xi32>) -> tensor<16x32xi32>

// -----

#map_ta = affine_map<(d0, d1, d2) -> (d2, d0)>
#map_tb_base = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_tc_base = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @contract_to_matmul_transpose_a(%arg0: tensor<8x16xf32>,
    %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.contract indexing_maps = [#map_ta, #map_tb_base, #map_tc_base]
    ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x32xf32>)
    outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-DAG: #[[$MAP_TA:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[$MAP_TB_BASE:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_TC_BASE:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @contract_to_matmul_transpose_a
// CHECK-SAME: %[[A:.+]]: tensor<8x16xf32>, %[[B:.+]]: tensor<8x32xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NOT: linalg.contract
// CHECK: linalg.matmul
// CHECK-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_TB_BASE]], #[[$MAP_TC_BASE]]]
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<8x16xf32>, tensor<8x32xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<16x32xf32>) -> tensor<16x32xf32>

// -----

#batch_map_a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#batch_map_b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#batch_map_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @contract_to_batch_matmul(%arg0: tensor<2x16x8xf32>,
    %arg1: tensor<2x8x16xf32>, %arg2: tensor<2x16x16xf32>)
    -> tensor<2x16x16xf32> {
  %0 = linalg.contract indexing_maps = [#batch_map_a, #batch_map_b, #batch_map_c]
    ins(%arg0, %arg1 : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
    outs(%arg2 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// CHECK-LABEL: @contract_to_batch_matmul
// CHECK-SAME: %[[A:.+]]: tensor<2x16x8xf32>, %[[B:.+]]: tensor<2x8x16xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK-NOT: linalg.contract
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// -----

#batch_map_ta = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#batch_map_tb = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#batch_map_tc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @contract_to_batch_matmul_transpose_a(%arg0: tensor<2x8x16xf32>,
    %arg1: tensor<2x8x32xf32>, %arg2: tensor<2x16x32xf32>)
    -> tensor<2x16x32xf32> {
  %0 = linalg.contract indexing_maps = [#batch_map_ta, #batch_map_tb, #batch_map_tc]
    ins(%arg0, %arg1 : tensor<2x8x16xf32>, tensor<2x8x32xf32>)
    outs(%arg2 : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %0 : tensor<2x16x32xf32>
}

// CHECK-DAG: #[[$BATCH_MAP_TA:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// CHECK-DAG: #[[$BATCH_MAP_TB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[$BATCH_MAP_TC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: @contract_to_batch_matmul_transpose_a
// CHECK-SAME: %[[A:.+]]: tensor<2x8x16xf32>, %[[B:.+]]: tensor<2x8x32xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.contract
// CHECK: linalg.batch_matmul
// CHECK-SAME: indexing_maps = [#[[$BATCH_MAP_TA]], #[[$BATCH_MAP_TB]], #[[$BATCH_MAP_TC]]]
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<2x8x16xf32>, tensor<2x8x32xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>

// -----

#non_identity_batch_a = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#non_identity_batch_b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#non_identity_batch_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// this stays as `linalg.contract` because the batch dimension is
// not in identity position across the operand/result maps. Named
// `linalg.batch_matmul` does not model such non-identity batch permutations.
func.func @contract_non_identity_batch(%arg0: tensor<4x2x8xf32>,
    %arg1: tensor<2x8x16xf32>, %arg2: tensor<2x4x16xf32>)
    -> tensor<2x4x16xf32> {
  %0 = linalg.contract indexing_maps = [#non_identity_batch_a, #non_identity_batch_b, #non_identity_batch_c]
    ins(%arg0, %arg1 : tensor<4x2x8xf32>, tensor<2x8x16xf32>)
    outs(%arg2 : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// CHECK-DAG: #[[$NON_ID_BATCH_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK-DAG: #[[$NON_ID_BATCH_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[$NON_ID_BATCH_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: @contract_non_identity_batch
// CHECK-SAME: %[[A:.+]]: tensor<4x2x8xf32>, %[[B:.+]]: tensor<2x8x16xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[$NON_ID_BATCH_A]], #[[$NON_ID_BATCH_B]], #[[$NON_ID_BATCH_C]]]
// CHECK-NOT: linalg.batch_matmul

// -----

#map_d = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_e = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
#map_f = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// this stays as `linalg.contract` because it has two reduction
// dimensions. Named matmul-like ops require exactly one M dim, one N dim, and
// one K dim.
func.func @contract_multi_reduction(%arg0: tensor<10x20x30xf32>,
    %arg1: tensor<30x20x40xf32>,
    %arg2: tensor<10x40xf32>) -> tensor<10x40xf32> {
  %0 = linalg.contract indexing_maps = [#map_d, #map_e, #map_f]
    ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
    outs(%arg2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %0 : tensor<10x40xf32>
}

// CHECK-LABEL: @contract_multi_reduction
// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}]
