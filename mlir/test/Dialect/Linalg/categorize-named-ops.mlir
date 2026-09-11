// RUN: mlir-opt %s -split-input-file -linalg-categorize-ops | FileCheck %s
// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=named-to-category | FileCheck %s -check-prefix=NAMED
// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=generic-to-category | FileCheck %s -check-prefix=GENERIC

func.func @categorize_matmul_buffer(%A : memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
  linalg.matmul ins(%A, %B: memref<16x8xf32>, memref<8x32xf32>)
               outs(%C: memref<16x32xf32>)
  return
}

// CHECK-DAG: #[[A_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[B_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @categorize_matmul_buffer
// CHECK-SAME: %[[A:.+]]: memref<16x8xf32>
// CHECK-SAME: %[[B:.+]]: memref<8x32xf32>
// CHECK-SAME: %[[C:.+]]: memref<16x32xf32>

// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<16x8xf32>, memref<8x32xf32>)
// CHECK-SAME: outs(%[[C]] : memref<16x32xf32>)

// NAMED-LABEL: func @categorize_matmul_buffer
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_matmul_buffer
// GENERIC-NOT: linalg.contract

// -----

func.func @categorize_matmul_tensor(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-DAG: #[[A_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[B_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @categorize_matmul_tensor

// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<16x8xf32>, tensor<8x32xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16x32xf32>)

// NAMED-LABEL: func @categorize_matmul_tensor
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_matmul_tensor
// GENERIC-NOT: linalg.contract

// -----

// Transpose A: A is accessed as (k, m) instead of (m, k).
func.func @categorize_matmul_transpose_a(%A: tensor<8x16xf32>, %B: tensor<8x32xf32>,
                                         %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B: tensor<8x16xf32>, tensor<8x32xf32>)
      outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-DAG: #[[TA_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[B_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @categorize_matmul_transpose_a
// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[TA_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<8x16xf32>, tensor<8x32xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16x32xf32>)

// NAMED-LABEL: func @categorize_matmul_transpose_a
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_matmul_transpose_a
// GENERIC-NOT: linalg.contract

// -----

// Broadcast B: B is accessed as (k) only, broadcasting over n.
func.func @categorize_matmul_broadcast_b(%A: tensor<16x8xf32>, %B: tensor<8xf32>,
                                         %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B: tensor<16x8xf32>, tensor<8xf32>)
      outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-DAG: #[[A_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[BB_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @categorize_matmul_broadcast_b
// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[BB_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<16x8xf32>, tensor<8xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16x32xf32>)

// NAMED-LABEL: func @categorize_matmul_broadcast_b
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_matmul_broadcast_b
// GENERIC-NOT: linalg.contract

// -----

// Transpose and broadcast A: A is accessed as (k) only, broadcasting over m.
func.func @categorize_matmul_transpose_broadcast_a(%A: tensor<8xf32>, %B: tensor<8x32xf32>,
                                                   %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B: tensor<8xf32>, tensor<8x32xf32>)
      outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-DAG: #[[TBA_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #[[B_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @categorize_matmul_transpose_broadcast_a
// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[TBA_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<8xf32>, tensor<8x32xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16x32xf32>)

// NAMED-LABEL: func @categorize_matmul_transpose_broadcast_a
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_matmul_transpose_broadcast_a
// GENERIC-NOT: linalg.contract

// -----

// Unsigned cast is preserved on the category op.
func.func @categorize_matmul_unsigned_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi16>,
                                           %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
                     ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi16>)
                     outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: func @categorize_matmul_unsigned_cast
// CHECK-NOT: linalg.matmul
// CHECK: linalg.contract
// CHECK-SAME: {cast = #linalg.type_fn<cast_unsigned>}

// NAMED-LABEL: func @categorize_matmul_unsigned_cast
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_matmul_unsigned_cast
// GENERIC-NOT: linalg.contract

// -----

func.func @categorize_batch_matmul(%A: tensor<2x3x5xf32>, %B: tensor<2x5x7xf32>, %C: tensor<2x3x7xf32>) -> tensor<2x3x7xf32> {
  %0 = linalg.batch_matmul ins(%A, %B: tensor<2x3x5xf32>, tensor<2x5x7xf32>)
                          outs(%C: tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
  return %0 : tensor<2x3x7xf32>
}

// CHECK-DAG: #[[A_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[B_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK: func @categorize_batch_matmul

// CHECK-NOT: linalg.batch_matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x3x5xf32>, tensor<2x5x7xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<2x3x7xf32>)

// NAMED-LABEL: func @categorize_batch_matmul
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_batch_matmul
// GENERIC-NOT: linalg.contract

// -----

func.func @categorize_batch_reduce_matmul(%A: memref<7x8x9xf32>, %B: memref<7x9x8xf32>, %C: memref<8x8xf32>) {
  linalg.batch_reduce_matmul ins(%A, %B: memref<7x8x9xf32>, memref<7x9x8xf32>)
                             outs(%C: memref<8x8xf32>)
  return
}

// CHECK-DAG: #[[A_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[B_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[C_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK: func @categorize_batch_reduce_matmul

// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<7x8x9xf32>, memref<7x9x8xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<8x8xf32>)

// NAMED-LABEL: func @categorize_batch_reduce_matmul
// NAMED: linalg.contract

// GENERIC-LABEL: func @categorize_batch_reduce_matmul
// GENERIC-NOT: linalg.contract

// -----

///----------------------------------------------------------------------------------------
/// Generic contraction ops (from specialize-generic-ops.mlir) also categorize
/// to linalg.contract.
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

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @op_matmul
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<?x?xf32>)

// GENERIC-LABEL: func @op_matmul
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_matmul
// NAMED-NOT: linalg.contract

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

// CHECK-DAG: #[[$MAP_TA:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @op_matmul_transpose_a
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[$MAP_TA]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<?x?xf32>)

// GENERIC-LABEL: func @op_matmul_transpose_a
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_matmul_transpose_a
// NAMED-NOT: linalg.contract

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

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_TB:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @op_matmul_transpose_b
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_TB]], #[[$MAP_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<?x?xf32>)

// GENERIC-LABEL: func @op_matmul_transpose_b
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_matmul_transpose_b
// NAMED-NOT: linalg.contract

// -----

#mapbA = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#mapbB = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#mapbC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul(%A: tensor<2x16x8xf32>, %B: tensor<2x8x16xf32>,
                           %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
    {indexing_maps = [#mapbA, #mapbB, #mapbC],
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

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func @op_batch_matmul
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x16x8xf32>, tensor<2x8x16xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<2x16x16xf32>)

// GENERIC-LABEL: func @op_batch_matmul
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_batch_matmul
// NAMED-NOT: linalg.contract

// -----

// A multi-reduction contraction (not a named-op matmul, still categorizes).
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

// CHECK-LABEL: func @op_multi_reduction
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract

// GENERIC-LABEL: func @op_multi_reduction
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_multi_reduction
// NAMED-NOT: linalg.contract

// -----

// TODO: named matvec.
#mapmvA = affine_map<(d0, d1) -> (d0, d1)>
#mapmvB = affine_map<(d0, d1) -> (d1)>
#mapmvC = affine_map<(d0, d1) -> (d0)>
func.func @op_matvec(%A: tensor<?x?xf32>, %B: tensor<?xf32>, %Out: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#mapmvA, #mapmvB, #mapmvC],
    iterator_types = ["parallel", "reduction"]}
    ins(%A, %B : tensor<?x?xf32>, tensor<?xf32>)
    outs(%Out : tensor<?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @op_matvec
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract

// GENERIC-LABEL: func @op_matvec
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_matvec
// NAMED-NOT: linalg.contract

// -----

#mapmA = affine_map<(m, n, k, m0, n0, k0) -> (m, k, m0, k0)>
#mapmB = affine_map<(m, n, k, m0, n0, k0) -> (n, k, n0, k0)>
#mapmC = affine_map<(m, n, k, m0, n0, k0) -> (m, n, m0, n0)>
func.func @op_mmt4d(%A: tensor<?x?x?x?xf32>, %B: tensor<?x?x?x?xf32>,
                    %C: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.generic
    {indexing_maps = [#mapmA, #mapmB, #mapmC],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs(%C : tensor<?x?x?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @op_mmt4d
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract

// GENERIC-LABEL: func @op_mmt4d
// GENERIC: linalg.contract

// NAMED-LABEL: func @op_mmt4d
// NAMED-NOT: linalg.contract
