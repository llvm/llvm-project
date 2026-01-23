// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#umap = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @unary_op_exp(%A: tensor<?x?x?xf32>, %Out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%A : tensor<?x?x?xf32>) outs(%Out : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = math.exp %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: unary_op_exp
// CHECK-SAME: %[[A:.+]]: tensor<?x?x?xf32>, %[[Out:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp ins(%[[A]] : tensor<?x?x?xf32>) outs(%[[Out]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @binary_op_div(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.divf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: binary_op_div
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,  %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.matmul
///----------------------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) {
   ^bb0(%in: f32, %in_0: f32, %out: f32):
     %1 = arith.mulf %in, %in_0 : f32
     %2 = arith.addf %out, %1 : f32
     linalg.yield %2 : f32
   } -> tensor<?x?xf32>
   return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: op_matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,  %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// Cast-auditing tests: ensure we only specialize when the cast semantics can
// be expressed by linalg.matmul, and use the cast attribute when needed.

// Check matmul with unsigned cast is correctly raised back to named op.
func.func @op_matmul_unsigned_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi32>,
                                   %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi32>) outs(%Out : tensor<16x32xi32>) {
  ^bb0(%in: i16, %in_0: i32, %out: i32):
    %1 = arith.extui %in : i16 to i32
    %3 = arith.muli %1, %in_0 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// CHECK-LABEL: op_matmul_unsigned_cast
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

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
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi64>) outs(%Out : tensor<16x32xi32>) {
  ^bb0(%in: i16, %in_0: i64, %out: i32):
    %1 = arith.extui %in : i16 to i32
    %2 = arith.trunci %in_0 : i64 to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %out, %3 : i32
    linalg.yield %4 : i32
  } -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// CHECK-LABEL: op_matmul_unsigned_cast_and_truncate
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

// Signed casts are the default, no cast attribute is required.
func.func @op_matmul_signed_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                 %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>) outs(%Out : tensor<16x32xi32>) {
   ^bb0(%in: i16, %in_0: i16, %out: i32):
     %1 = arith.extsi %in : i16 to i32
     %2 = arith.extsi %in_0 : i16 to i32
     %3 = arith.muli %1, %2 : i32
     %4 = arith.addi %out, %3 : i32
     linalg.yield %4 : i32
   } -> tensor<16x32xi32>
   return %0 : tensor<16x32xi32>
}

// CHECK-LABEL: op_matmul_signed_cast
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
// CHECK: linalg.matmul

// Mixed signed/unsigned inputs cannot be encoded with a single cast attribute.
func.func @negative_op_matmul_mixed_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>) outs(%Out : tensor<16x32xi32>) {
   ^bb0(%in: i16, %in_0: i16, %out: i32):
     %1 = arith.extui %in : i16 to i32
     %2 = arith.extsi %in_0 : i16 to i32
     %3 = arith.muli %1, %2 : i32
     %4 = arith.addi %out, %3 : i32
     linalg.yield %4 : i32
   } -> tensor<16x32xi32>
   return %0 : tensor<16x32xi32>
}

// CHECK-LABEL: negative_op_matmul_mixed_cast
// CHECK: linalg.generic
// CHECK-NOT: linalg.matmul

// Output-side casts are not representable by the named matmul ops.
func.func @negative_op_matmul_output_cast(%A: tensor<16x8xi32>, %B: tensor<8x32xi32>,
                                 %Out: tensor<16x32xi64>) -> tensor<16x32xi64> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi32>, tensor<8x32xi32>) outs(%Out : tensor<16x32xi64>) {
   ^bb0(%in: i32, %in_0: i32, %out: i64):
     %3 = arith.trunci %out : i64 to i32
     %4 = arith.muli %in, %in_0 : i32
     %5 = arith.addi %3, %4 : i32
     %6 = arith.extsi %5 : i32 to i64
     linalg.yield %6 : i64
   } -> tensor<16x32xi64>
   return %0 : tensor<16x32xi64>
}

// CHECK-LABEL: negative_op_matmul_output_cast
// CHECK: linalg.generic
// CHECK-NOT: linalg.matmul

// Bitcasts are not modeled by the cast attribute, but should not block
// specialization.
// NOTE: Bitcasts are not preserved by the matmul named op during
// roundtrip, so this is potentially loosing information here.
// See #177593 for more details.
func.func @op_matmul_bitcast_int_to_float(%A: tensor<16x8xi32>,
                                          %B: tensor<8x32xi32>,
                                          %Out: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi32>, tensor<8x32xi32>) outs(%Out : tensor<16x32xf32>) {
  ^bb0(%in: i32, %in_0: i32, %out: f32):
    %1 = arith.bitcast %in : i32 to f32
    %2 = arith.bitcast %in_0 : i32 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: op_matmul_bitcast_int_to_float
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul

// Signed float casts only use sitofp, which defaults to signed semantics.
func.func @op_matmul_signed_cast_float(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                       %Out: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>) outs(%Out : tensor<16x32xf32>) {
  ^bb0(%in: i16, %in_0: i16, %out: f32):
    %1 = arith.sitofp %in : i16 to f32
    %2 = arith.sitofp %in_0 : i16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: op_matmul_signed_cast_float
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
// CHECK: linalg.matmul

// Unsigned float casts are expressed via uitofp and use the unsigned cast attr.
func.func @op_matmul_unsigned_cast_float(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                         %Out: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi16>) outs(%Out : tensor<16x32xf32>) {
  ^bb0(%in: i16, %in_0: i16, %out: f32):
    %1 = arith.uitofp %in : i16 to f32
    %2 = arith.uitofp %in_0 : i16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: op_matmul_unsigned_cast_float
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.batch_matmul
///----------------------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul(%A: tensor<2x16x8xf32>, %B: tensor<2x8x16xf32>, %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
           {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
           ins(%A, %B : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// CHECK-LABEL: op_batch_matmul
// CHECK-SAME: %[[A:.+]]: tensor<2x16x8xf32>, %[[B:.+]]: tensor<2x8x16xf32>,  %[[Out:.+]]: tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_matmul ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%[[Out]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

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

// CHECK-LABEL: op_batch_matmul_unsigned_cast
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_matmul {cast = #linalg.type_fn<cast_unsigned>}

// -----

// This is a multi-reduction linalg.generic and cannot be lifted to matrix multiply
#mapA = affine_map<(m, n, k1, k2) -> (m, k1, k2)>
#mapB = affine_map<(m, n, k1, k2) -> (k2, k1, n)>
#mapC = affine_map<(m, n, k1, k2) -> (m, n)>
func.func @negative_op_multi_reduction(%A: tensor<10x20x30xf32>,
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

// CHECK-LABEL: negative_op_multi_reduction
// CHECK: linalg.generic

// -----

// TODO: matvec
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
// CHECK-LABEL: op_matvec
// CHECK: linalg.generic
