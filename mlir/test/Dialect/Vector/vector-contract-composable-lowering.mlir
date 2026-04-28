// RUN: mlir-opt %s --test-vector-contract-lowering-composition="mode=dot-outerproduct" --split-input-file | FileCheck %s --check-prefix=DOT
// RUN: mlir-opt %s --test-vector-contract-lowering-composition="mode=generic" --split-input-file | FileCheck %s --check-prefix=GENERIC
// RUN: mlir-opt %s --test-vector-contract-lowering-composition="mode=parallel-arith-reject" --split-input-file | FileCheck %s --check-prefix=PARALLEL

#matmat_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// DOT-LABEL: func @dot_accept
// DOT-NOT: vector.outerproduct
// DOT: vector.reduction <add>
func.func @dot_accept(%A: vector<2x4xf32>,
                      %B: vector<4x3xf32>,
                      %C: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %A, %B, %C
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// DOT-LABEL: func @dot_reject_to_outerproduct
// DOT: vector.outerproduct
func.func @dot_reject_to_outerproduct(%A: vector<2x4xf32>,
                                      %B: vector<4x3xf32>,
                                      %C: vector<2x3xf32>)
                                      -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %A, %B, %C
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// -----

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_add_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}
#dotp_mul_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"],
  kind = #vector.kind<mul>
}

// GENERIC-LABEL: func @generic_add
// GENERIC: arith.mulf
// GENERIC: vector.reduction <add>
func.func @generic_add(%A: vector<4xf32>, %B: vector<4xf32>,
                       %C: f32) -> f32 {
  %0 = vector.contract #dotp_add_trait %A, %B, %C
    : vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
}

// GENERIC-LABEL: func @generic_non_add
// GENERIC: vector.contract
// GENERIC-SAME: kind = #vector.kind<mul>
func.func @generic_non_add(%A: vector<4xf32>, %B: vector<4xf32>,
                           %C: f32) -> f32 {
  %0 = vector.contract #dotp_mul_trait %A, %B, %C
    : vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
}

// -----

// PARALLEL-LABEL: func @parallel_arith_filter_reject
// PARALLEL: vector.contract
func.func @parallel_arith_filter_reject(
    %A: vector<1x1x4xf32>, %B: vector<1x1x4xf32>,
    %C: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
      affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
      affine_map<(d0, d1, d2) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction", "reduction"],
    kind = #vector.kind<add>
  } %A, %B, %C : vector<1x1x4xf32>, vector<1x1x4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}
