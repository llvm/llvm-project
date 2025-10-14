// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,64 \
// RUN: lhs-transpose-outer-blocks=false lhs-transpose-inner-blocks=false \
// RUN: rhs-transpose-outer-blocks=true rhs-transpose-inner-blocks=true" \
// RUN: -canonicalize | FileCheck %s --check-prefix=MMT4D

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,64 \
// RUN: lhs-transpose-outer-blocks=false lhs-transpose-inner-blocks=false \
// RUN: rhs-transpose-outer-blocks=false rhs-transpose-inner-blocks=false" \
// RUN: -canonicalize | FileCheck %s --check-prefix=MM4D

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,64 \
// RUN: lhs-transpose-outer-blocks=true lhs-transpose-inner-blocks=true \
// RUN: rhs-transpose-outer-blocks=false rhs-transpose-inner-blocks=false" \
// RUN: -canonicalize | FileCheck %s --check-prefix=MTM4D

func.func @block_matmul(
    %A: tensor<64x128xf32>, %B: tensor<128x64xf32>, %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul  ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
                      outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// MMT4D-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MMT4D-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// MMT4D-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MMT4D-LABEL: func @block_matmul
// MMT4D-COUNT-3: linalg.pack
// MMT4D: linalg.generic
// MMT4D-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// MMT4D-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// MMT4D-COUNT-1: linalg.unpack

// MM4D-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MM4D-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
// MM4D-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MM4D-LABEL: func @block_matmul
// MM4D-COUNT-3: linalg.pack
// MM4D: linalg.generic
// MM4D-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// MM4D-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// MM4D-COUNT-1: linalg.unpack

// MTM4D-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// MTM4D-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
// MTM4D-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MTM4D-LABEL: func @block_matmul
// MTM4D-COUNT-3: linalg.pack
// MTM4D: linalg.generic
// MTM4D-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// MTM4D-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// MTM4D-COUNT-1: linalg.unpack
