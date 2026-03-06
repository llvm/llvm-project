// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=generic-to-named \
// RUN: | FileCheck %s --check-prefix=NAMED,ALL

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=generic-to-category \
// RUN: | FileCheck %s --check-prefix=CATEGORY,ALL

#umap = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @unary_op_exp(%A: tensor<?x?x?xf32>, %Out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#umap, #umap],
    iterator_types = ["parallel", "parallel","parallel"]}
    ins(%A : tensor<?x?x?xf32>)
    outs(%Out : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = math.exp %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// ALL-LABEL: unary_op_exp
// ALL-SAME: %[[A:.+]]: tensor<?x?x?xf32>, %[[OUT:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.exp
// NAMED-SAME: ins(%[[A]] : tensor<?x?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

// Not supported yet.
// CATEGORY: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @binary_op_div(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                         %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.divf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// ALL-LABEL: binary_op_div
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// ALL-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.div
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// Not supported yet.
// CATEGORY: linalg.generic

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.matmul
///----------------------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                     %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CATEGORY-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// ALL-LABEL: op_matmul
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// ALL-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// Cast-auditing tests: ensure we only specialize when the cast semantics can
// be expressed by linalg.matmul, and use the cast attribute when needed.

// Check matmul with unsigned cast is correctly raised back to named op.
func.func @op_matmul_unsigned_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi32>,
                                   %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi32>)
    outs(%Out : tensor<16x32xi32>) {
  ^bb0(%in: i16, %in_0: i32, %out: i32):
    %1 = arith.extui %in : i16 to i32
    %3 = arith.muli %1, %in_0 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// ALL-LABEL: op_matmul_unsigned_cast

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CATEGORY-SAME: {cast = #linalg.type_fn<cast_unsigned>}

// Ensures truncation rounding is tolerated with unsigned cases.
// Note: We only consider casts as conflicting if they have different
// signedness behaviours, and then we do not specialize if they do
// conflict. Since this is not such a case, we do not block specialization.
// Also the roundtrip lowering back to linalg.generic for such an op is
// expected to produce the same thing again, so we are not loosing
// information here.
func.func @op_matmul_unsigned_cast_and_truncate(%A: tensor<16x8xi16>, %B: tensor<8x32xi64>,
                                                %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi64>)
    outs(%Out : tensor<16x32xi32>) {
  ^bb0(%in: i16, %in_0: i64, %out: i32):
    %1 = arith.extui %in : i16 to i32
    %2 = arith.trunci %in_0 : i64 to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// ALL-LABEL: op_matmul_unsigned_cast_and_truncate

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CATEGORY-SAME: {cast = #linalg.type_fn<cast_unsigned>}

// Signed casts are the default, no cast attribute is required.
func.func @op_matmul_signed_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                 %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>)
    outs(%Out : tensor<16x32xi32>) {
  ^bb0(%in: i16, %in_0: i16, %out: i32):
    %1 = arith.extsi %in : i16 to i32
    %2 = arith.extsi %in_0 : i16 to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// ALL-LABEL: op_matmul_signed_cast

// NAMED-NOT: linalg.generic
// NAMED-NOT: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
// NAMED: linalg.matmul

// CATEGORY-NOT: linalg.generic
// CATEGORY: %[[RES:.+]] = linalg.contract
// CATEGORY-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CATEGORY-NOT: {cast =
// CATEGORY-SAME: ins
// CATEGORY: return %[[RES]]

// Mixed signed/unsigned inputs cannot be encoded with a single cast attribute.
func.func @negative_op_matmul_mixed_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                         %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>)
    outs(%Out : tensor<16x32xi32>) {
  ^bb0(%in: i16, %in_0: i16, %out: i32):
    %1 = arith.extui %in : i16 to i32
    %2 = arith.extsi %in_0 : i16 to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// ALL-LABEL: negative_op_matmul_mixed_cast

// NAMED: linalg.generic
// NAMED-NOT: linalg.matmul

// CATEGORY: linalg.generic
// CATEGORY-NOT: linalg.contract

// Output-side casts are not representable by the named matmul ops.
func.func @negative_op_matmul_output_cast(%A: tensor<16x8xi32>, %B: tensor<8x32xi32>,
                                          %Out: tensor<16x32xi64>) -> tensor<16x32xi64> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi32>, tensor<8x32xi32>)
    outs(%Out : tensor<16x32xi64>) {
  ^bb0(%in: i32, %in_0: i32, %out: i64):
    %3 = arith.trunci %out : i64 to i32
    %4 = arith.muli %in, %in_0 : i32
    %5 = arith.addi %3, %4 : i32
    %6 = arith.extsi %5 : i32 to i64
    linalg.yield %6 : i64
  } -> tensor<16x32xi64>
  return %0 : tensor<16x32xi64>
}

// ALL-LABEL: negative_op_matmul_output_cast

// NAMED: linalg.generic
// NAMED-NOT: linalg.matmul

// CATEGORY: linalg.generic
// CATEGORY-NOT: linalg.contract

// Bitcasts are not modeled by the cast attribute, but should not block
// specialization.
// NOTE: Bitcasts are not preserved by the matmul named op during
// roundtrip, so this is potentially loosing information here.
// See #177593 for more details.
func.func @op_matmul_bitcast_int_to_float(%A: tensor<16x8xi32>,
                                          %B: tensor<8x32xi32>,
                                          %Out: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi32>, tensor<8x32xi32>)
    outs(%Out : tensor<16x32xf32>) {
  ^bb0(%in: i32, %in_0: i32, %out: f32):
    %1 = arith.bitcast %in : i32 to f32
    %2 = arith.bitcast %in_0 : i32 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// ALL-LABEL: op_matmul_bitcast_int_to_float

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract

// Signed float casts only use sitofp, which defaults to signed semantics.
func.func @op_matmul_signed_cast_float(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                       %Out: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>)
    outs(%Out : tensor<16x32xf32>) {
  ^bb0(%in: i16, %in_0: i16, %out: f32):
    %1 = arith.sitofp %in : i16 to f32
    %2 = arith.sitofp %in_0 : i16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// ALL-LABEL: op_matmul_signed_cast_float

// NAMED-NOT: linalg.generic
// NAMED-NOT: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
// NAMED: linalg.matmul

// CATEGORY-NOT: linalg.generic
// CATEGORY-NOT: linalg.contract{{.*}}{cast =
// CATEGORY: %[[RES:.+]] = linalg.contract
// CATEGORY: return %[[RES]]

// Unsigned float casts are expressed via uitofp and use the unsigned cast attr.
func.func @op_matmul_unsigned_cast_float(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                         %Out: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>)
    outs(%Out : tensor<16x32xf32>) {
  ^bb0(%in: i16, %in_0: i16, %out: f32):
    %1 = arith.uitofp %in : i16 to f32
    %2 = arith.uitofp %in_0 : i16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// ALL-LABEL: op_matmul_unsigned_cast_float

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract{{.*}}{cast = #linalg.type_fn<cast_unsigned>}

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.batch_matmul
///----------------------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul(%A: tensor<2x16x8xf32>, %B: tensor<2x8x16xf32>,
                           %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
    outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CATEGORY-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// ALL-LABEL: op_batch_matmul
// ALL-SAME: %[[A:.+]]: tensor<2x16x8xf32>, %[[B:.+]]: tensor<2x8x16xf32>,  %[[OUT:.+]]: tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.batch_matmul ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// Ensure that the unsigned cast path for cast detection is exercised for
// batch_matmul as well.
func.func @op_batch_matmul_unsigned_cast(%A: tensor<2x16x8xi16>,
                                         %B: tensor<2x8x16xi64>,
                                         %Out: tensor<2x16x16xi32>) -> tensor<2x16x16xi32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<2x16x8xi16>, tensor<2x8x16xi64>)
    outs(%Out : tensor<2x16x16xi32>) {
  ^bb0(%in: i16, %in_0: i64, %out: i32):
    %1 = arith.extui %in : i16 to i32
    %2 = arith.trunci %in_0 : i64 to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<2x16x16xi32>
  return %0 : tensor<2x16x16xi32>
}

// ALL-LABEL: op_batch_matmul_unsigned_cast

// NAMED-NOT: linalg.generic
// NAMED: linalg.batch_matmul {cast = #linalg.type_fn<cast_unsigned>}

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CATEGORY-SAME: {cast = #linalg.type_fn<cast_unsigned>}

// -----

// A multi-reduction contraction.
#mapA = affine_map<(m, n, k1, k2) -> (m, k1, k2)>
#mapB = affine_map<(m, n, k1, k2) -> (k2, k1, n)>
#mapC = affine_map<(m, n, k1, k2) -> (m, n)>
func.func @op_multi_reduction(%A: tensor<10x20x30xf32>,
                              %B: tensor<30x20x40xf32>,
                              %C: tensor<10x40xf32>) -> tensor<10x40xf32> {
  %0 = linalg.generic
    {indexing_maps = [#mapA, #mapB, #mapC],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%A, %B : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
    outs(%C : tensor<10x40xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %1 = arith.mulf %a, %b : f32
    %2 = arith.addf %c, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<10x40xf32>
  return %0 : tensor<10x40xf32>
}

// ALL-LABEL: op_multi_reduction

// Cannot be lifted to named matrix multiply.
// NAMED: linalg.generic

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract

// -----

// Batch dim not in identity position: batch dim d0 appears at result
// position 1 in A's map instead of position 0.
#mapBni0 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#mapBni1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#mapBni2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_non_identity_batch(%A: tensor<4x2x8xf32>, %B: tensor<2x8x16xf32>,
                                           %Out: tensor<2x4x16xf32>) -> tensor<2x4x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#mapBni0, #mapBni1, #mapBni2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<4x2x8xf32>, tensor<2x8x16xf32>)
    outs(%Out : tensor<2x4x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// ALL-LABEL: batch_matmul_non_identity_batch

// Cannot be lifted to named matrix multiply.
// NAMED: linalg.generic

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract

// -----

// TODO: named matvec
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @op_matvec(%A: tensor<?x?xf32>, %B: tensor<?xf32>, %Out: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?xf32>) outs(%Out : tensor<?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// ALL-LABEL: op_matvec

// NAMED: linalg.generic

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract

// -----

// Matmul transpose A: A is accessed as (k, m) instead of (m, k)
#map_ta = affine_map<(d0, d1, d2) -> (d2, d0)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul_transpose_a(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                 %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_ta, #map_b, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// ALL-DAG: #[[$MAP_TA:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// ALL-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// ALL-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// ALL-LABEL: op_matmul_transpose_a
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_B]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_B]], #[[$MAP_C]]]

// -----

// Matmul transpose B: B is accessed as (n, k) instead of (k, n)
#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_tb = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul_transpose_b(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                 %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_a, #map_tb, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// ALL-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// ALL-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// ALL-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// ALL-LABEL: op_matmul_transpose_b
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// Batch matmul transpose A: A is accessed as (b, k, m) instead of (b, m, k)
#map_ta = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map_b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul_transpose_a(%A: tensor<2x8x4xf32>, %B: tensor<2x8x16xf32>,
                                       %Out: tensor<2x4x16xf32>) -> tensor<2x4x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_ta, #map_b, #map_c],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<2x8x4xf32>, tensor<2x8x16xf32>)
    outs(%Out : tensor<2x4x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// ALL-DAG: #[[$MAP_TA:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// ALL-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// ALL-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// ALL-LABEL: op_batch_matmul_transpose_a
// ALL-SAME: %[[A:.+]]: tensor<2x8x4xf32>, %[[B:.+]]: tensor<2x8x16xf32>, %[[OUT:.+]]: tensor<2x4x16xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.batch_matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_B]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<2x8x4xf32>, tensor<2x8x16xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_B]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<2x8x4xf32>, tensor<2x8x16xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>

// -----

// Batch matmul transpose B: B is accessed as (b, n, k) instead of (b, k, n)
#map_a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_tb = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul_transpose_b(%A: tensor<2x4x8xf32>, %B: tensor<2x16x8xf32>,
                                       %Out: tensor<2x4x16xf32>) -> tensor<2x4x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_a, #map_tb, #map_c],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<2x4x8xf32>, tensor<2x16x8xf32>)
    outs(%Out : tensor<2x4x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// ALL-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// ALL-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// ALL-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// ALL-LABEL: op_batch_matmul_transpose_b
// ALL-SAME: %[[A:.+]]: tensor<2x4x8xf32>, %[[B:.+]]: tensor<2x16x8xf32>, %[[OUT:.+]]: tensor<2x4x16xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.batch_matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<2x4x8xf32>, tensor<2x16x8xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<2x4x8xf32>, tensor<2x16x8xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>

// -----

// Both A and B transposed.
#map_ta = affine_map<(d0, d1, d2) -> (d2, d0)>
#map_tb = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul_transpose_a_and_b(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                       %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_ta, #map_tb, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// ALL-DAG: #[[$MAP_TA:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// ALL-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// ALL-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// ALL-LABEL: op_matmul_transpose_a_and_b
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_TB]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_TB]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// Output transposed: C is accessed as (n, m) instead of (m, n).
#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_tc = affine_map<(d0, d1, d2) -> (d1, d0)>
func.func @op_matmul_transposed_output(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                       %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_a, #map_b, #map_tc],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// ALL-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// ALL-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// ALL-DAG: #[[$MAP_TC:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// ALL-LABEL: op_matmul_transposed_output
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_TC]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_TC]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// Matmul with non-canonical loop ordering.
#map_nc_a = affine_map<(m, k, n) -> (m, k)>
#map_nc_b = affine_map<(m, k, n) -> (k, n)>
#map_nc_c = affine_map<(m, k, n) -> (m, n)>
func.func @op_matmul_non_canonical_loops(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                         %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_nc_a, #map_nc_b, #map_nc_c],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CATEGORY-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// ALL-LABEL: op_matmul_non_canonical_loops
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// -----

// Batch matmul with non-canonical loop ordering.
#map_bnc_a = affine_map<(batch, m, k, n) -> (batch, m, k)>
#map_bnc_b = affine_map<(batch, m, k, n) -> (batch, k, n)>
#map_bnc_c = affine_map<(batch, m, k, n) -> (batch, m, n)>
func.func @op_batch_matmul_non_canonical_loops(%A: tensor<2x16x8xf32>, %B: tensor<2x8x16xf32>,
                                               %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_bnc_a, #map_bnc_b, #map_bnc_c],
    iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
    ins(%A, %B : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
    outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CATEGORY-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

// ALL-LABEL: op_batch_matmul_non_canonical_loops
// ALL-SAME: %[[A:.+]]: tensor<2x16x8xf32>, %[[B:.+]]: tensor<2x8x16xf32>, %[[OUT:.+]]: tensor<2x16x16xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.batch_matmul
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// -----

// Matmul with non-canonical loop ordering (d0=m, d1=k, d2=n) and B transposed.
#map_nc_tb_a = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_nc_tb_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_nc_tb_c = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @op_matmul_non_canonical_transpose_b(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                               %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_nc_tb_a, #map_nc_tb_b, #map_nc_tb_c],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// NAMED-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// NAMED-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// NAMED-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CATEGORY-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// ALL-LABEL: op_matmul_non_canonical_transpose_b
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// Batch matmul with non-canonical loop ordering (d0=batch, d1=m, d2=k, d3=n)
// and B Transposed.
#map_bnc_tb_a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map_bnc_tb_b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map_bnc_tb_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
func.func @op_batch_matmul_non_canonical_transpose_b(%A: tensor<2x16x8xf32>, %B: tensor<2x16x8xf32>,
                                                     %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_bnc_tb_a, #map_bnc_tb_b, #map_bnc_tb_c],
    iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
    ins(%A, %B : tensor<2x16x8xf32>, tensor<2x16x8xf32>)
    outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// NAMED-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// NAMED-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// NAMED-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CATEGORY-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

// ALL-LABEL: op_batch_matmul_non_canonical_transpose_b
// ALL-SAME: %[[A:.+]]: tensor<2x16x8xf32>, %[[B:.+]]: tensor<2x16x8xf32>, %[[OUT:.+]]: tensor<2x16x16xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.batch_matmul
// NAMED-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x16x8xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x16x8xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// -----

// Matmul with fully permuted loop ordering.
#map_fs_a = affine_map<(d0, d1, d2) -> (d1, d0)>
#map_fs_b = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_fs_c = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @op_matmul_fully_shuffled_loops(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                          %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_fs_a, #map_fs_b, #map_fs_c],
    iterator_types = ["reduction", "parallel", "parallel"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CATEGORY-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CATEGORY-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CATEGORY-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// ALL-LABEL: op_matmul_fully_shuffled_loops
// ALL-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>

// NAMED-NOT: linalg.generic
// NAMED: linalg.matmul
// NAMED-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// NAMED-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
// CATEGORY-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// CATEGORY-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CATEGORY-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// TODO: this could also be specialized to a named matmul.
#map_bcast_a = affine_map<(d0, d1, d2) -> (d2)>
#map_bcast_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_bcast_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul_broadcast_a(%A: tensor<?xf32>, %B: tensor<?x?xf32>,
                                 %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_bcast_a, #map_bcast_b, #map_bcast_c],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// ALL-LABEL: op_matmul_broadcast_a

// NAMED: linalg.generic
// NAMED-NOT: linalg.matmul

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract

// -----

// TODO: this could also be specialized to a named batch_matmul.
#map_bbcast_a = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map_bbcast_b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map_bbcast_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul_broadcast_a(%A: tensor<16x8xf32>, %B: tensor<2x8x16xf32>,
                                       %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_bbcast_a, #map_bbcast_b, #map_bbcast_c],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<16x8xf32>, tensor<2x8x16xf32>)
    outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// ALL-LABEL: op_batch_matmul_broadcast_a

// NAMED: linalg.generic
// NAMED-NOT: linalg.batch_matmul

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract

// -----

// TODO: this could also be specialized to a named batch_matmul.
#map_bbcast2_a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_bbcast2_b = affine_map<(d0, d1, d2, d3) -> (d3)>
#map_bbcast2_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul_broadcast_b(%A: tensor<2x16x8xf32>, %B: tensor<8xf32>,
                                       %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#map_bbcast2_a, #map_bbcast2_b, #map_bbcast2_c],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<2x16x8xf32>, tensor<8xf32>)
    outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// ALL-LABEL: op_batch_matmul_broadcast_b

// NAMED: linalg.generic
// NAMED-NOT: linalg.batch_matmul

// CATEGORY-NOT: linalg.generic
// CATEGORY: linalg.contract
