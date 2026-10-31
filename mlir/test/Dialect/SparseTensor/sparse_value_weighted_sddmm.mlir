// RUN: mlir-opt %s --pre-sparsification-rewrite --sparse-reinterpret-map --sparsification="parallelization-strategy=dense-outer-loop" --cse | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#trait_value_weighted_sddmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,
    affine_map<(i,j,k) -> (i,k)>,
    affine_map<(i,j,k) -> (k,j)>,
    affine_map<(i,j,k) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

#trait_matmul = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,
    affine_map<(i,j,k) -> (k,j)>,
    affine_map<(i,j,k) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func.func @value_weighted_sddmm(
// CHECK-SAME:    %[[SAMPLE:[^:]+]]: tensor<4x5xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %{{[^:]+}}: tensor<4x3xf64>,
// CHECK-SAME:    %{{[^:]+}}: tensor<3x5xf64>)
// CHECK:         %[[POS:.*]] = sparse_tensor.positions %[[SAMPLE]] {level = 1 : index}
// CHECK:         %[[CRD:.*]] = sparse_tensor.coordinates %[[SAMPLE]] {level = 1 : index}
// CHECK:         %[[SAMPLE_VAL:.*]] = sparse_tensor.values %[[SAMPLE]]
// CHECK:         %[[POS_TENSOR:.*]] = bufferization.to_tensor %[[POS]]
// CHECK:         %[[POS_COPY:.*]] = bufferization.alloc_tensor() copy(%[[POS_TENSOR]])
// CHECK:         %[[CRD_TENSOR:.*]] = bufferization.to_tensor %[[CRD]]
// CHECK:         %[[CRD_COPY:.*]] = bufferization.alloc_tensor() copy(%[[CRD_TENSOR]])
// CHECK:         %[[RESULT_VAL_TENSOR:.*]] = bufferization.alloc_tensor(%{{.*}}) : tensor<?xf64>
// CHECK:         %[[ZEROED:.*]] = linalg.fill
// CHECK:         %[[RESULT:.*]] = sparse_tensor.assemble (%[[POS_COPY]], %[[CRD_COPY]]), %[[ZEROED]]
// CHECK:         %[[RESULT_VAL:.*]] = sparse_tensor.values %[[RESULT]]
// CHECK-NOT:     sparse_tensor.expand
// CHECK:         scf.parallel
// CHECK:           %[[ACC:.*]] = memref.load %[[RESULT_VAL]][%{{.*}}]
// CHECK:           %[[S:.*]] = memref.load %[[SAMPLE_VAL]][%{{.*}}]
// CHECK:           %[[AB:.*]] = arith.mulf
// CHECK:           %[[WEIGHTED:.*]] = arith.mulf %[[S]], %[[AB]] : f64
// CHECK:           %[[NEXT:.*]] = arith.addf %[[ACC]], %[[WEIGHTED]] : f64
// CHECK:           memref.store %[[NEXT]], %[[RESULT_VAL]][%{{.*}}]
// CHECK-NOT:     sparse_tensor.compress
// CHECK:         %[[LOADED:.*]] = sparse_tensor.load %[[RESULT]]
// CHECK:         return %[[LOADED]]
func.func @value_weighted_sddmm(
    %sample: tensor<4x5xf64, #CSR>,
    %lhs: tensor<4x3xf64>,
    %rhs: tensor<3x5xf64>) -> tensor<4x5xf64, #CSR> {
  %empty = tensor.empty() : tensor<4x5xf64, #CSR>
  %result = linalg.generic #trait_value_weighted_sddmm
      ins(%sample, %lhs, %rhs : tensor<4x5xf64, #CSR>,
                                  tensor<4x3xf64>, tensor<3x5xf64>)
      outs(%empty : tensor<4x5xf64, #CSR>) {
    ^bb0(%s: f64, %a: f64, %b: f64, %acc: f64):
      %ab = arith.mulf %a, %b : f64
      %weighted = arith.mulf %s, %ab : f64
      %next = arith.addf %acc, %weighted : f64
      linalg.yield %next : f64
  } -> tensor<4x5xf64, #CSR>
  return %result : tensor<4x5xf64, #CSR>
}

// Verify that the existing sparse multiply-over-add fusion produces a form
// accepted by the same structure-preserving lowering.
// CHECK-LABEL: func.func @value_weighted_sddmm_unfused(
// CHECK:         %[[SAMPLE_VAL:.*]] = sparse_tensor.values %[[SAMPLE:.*]]
// CHECK:         %[[RESULT:.*]] = sparse_tensor.assemble
// CHECK:         %[[RESULT_VAL:.*]] = sparse_tensor.values %[[RESULT]]
// CHECK-NOT:     sparse_tensor.expand
// CHECK:         scf.parallel
// CHECK:           %[[AB:.*]] = arith.mulf
// CHECK:           %[[S:.*]] = memref.load %[[SAMPLE_VAL]][%{{.*}}]
// CHECK:           %[[WEIGHTED:.*]] = arith.mulf
// CHECK:           memref.store %{{.*}}, %[[RESULT_VAL]][%{{.*}}]
// CHECK-NOT:     sparse_tensor.compress
func.func @value_weighted_sddmm_unfused(
    %sample: tensor<4x5xf64, #CSR>,
    %lhs: tensor<4x3xf64>,
    %rhs: tensor<3x5xf64>) -> tensor<4x5xf64, #CSR> {
  %zero = arith.constant dense<0.0> : tensor<4x5xf64>
  %product = linalg.generic #trait_matmul
      ins(%lhs, %rhs : tensor<4x3xf64>, tensor<3x5xf64>)
      outs(%zero : tensor<4x5xf64>) {
    ^bb0(%a: f64, %b: f64, %acc: f64):
      %ab = arith.mulf %a, %b : f64
      %next = arith.addf %acc, %ab : f64
      linalg.yield %next : f64
  } -> tensor<4x5xf64>
  %empty = tensor.empty() : tensor<4x5xf64, #CSR>
  %result = linalg.generic #trait_scale
      ins(%product, %sample : tensor<4x5xf64>, tensor<4x5xf64, #CSR>)
      outs(%empty : tensor<4x5xf64, #CSR>) {
    ^bb0(%productValue: f64, %sampleValue: f64, %unused: f64):
      %weighted = arith.mulf %productValue, %sampleValue : f64
      linalg.yield %weighted : f64
  } -> tensor<4x5xf64, #CSR>
  return %result : tensor<4x5xf64, #CSR>
}

// Structure reuse is derived from the tensor expression, rather than from an
// SDDMM-specific matcher.
// CHECK-LABEL: func.func @structure_reuse_elementwise(
// CHECK:         sparse_tensor.positions %[[SAMPLE:.*]]
// CHECK:         sparse_tensor.coordinates %[[SAMPLE]]
// CHECK:         sparse_tensor.assemble
// CHECK-NOT:     sparse_tensor.expand
// CHECK:         scf.parallel
// CHECK:           arith.mulf
// CHECK:           memref.store
// CHECK-NOT:     sparse_tensor.compress
func.func @structure_reuse_elementwise(
    %sample: tensor<4x5xf64, #CSR>,
    %factor: tensor<4x5xf64>) -> tensor<4x5xf64, #CSR> {
  %empty = tensor.empty() : tensor<4x5xf64, #CSR>
  %result = linalg.generic #trait_scale
      ins(%sample, %factor : tensor<4x5xf64, #CSR>, tensor<4x5xf64>)
      outs(%empty : tensor<4x5xf64, #CSR>) {
    ^bb0(%sampleValue: f64, %factorValue: f64, %unused: f64):
      %scaled = arith.mulf %sampleValue, %factorValue : f64
      linalg.yield %scaled : f64
  } -> tensor<4x5xf64, #CSR>
  return %result : tensor<4x5xf64, #CSR>
}

// A second sparse condition can only reduce the result support, so either
// operand provides a safe structure source.
// CHECK-LABEL: func.func @structure_reuse_intersection(
// CHECK:         sparse_tensor.assemble
// CHECK-NOT:     sparse_tensor.expand
// CHECK:         scf.parallel
// CHECK:           arith.mulf
// CHECK-NOT:     sparse_tensor.compress
func.func @structure_reuse_intersection(
    %lhs: tensor<4x5xf64, #CSR>,
    %rhs: tensor<4x5xf64, #CSR>) -> tensor<4x5xf64, #CSR> {
  %empty = tensor.empty() : tensor<4x5xf64, #CSR>
  %result = linalg.generic #trait_scale
      ins(%lhs, %rhs : tensor<4x5xf64, #CSR>, tensor<4x5xf64, #CSR>)
      outs(%empty : tensor<4x5xf64, #CSR>) {
    ^bb0(%lhsValue: f64, %rhsValue: f64, %unused: f64):
      %product = arith.mulf %lhsValue, %rhsValue : f64
      linalg.yield %product : f64
  } -> tensor<4x5xf64, #CSR>
  return %result : tensor<4x5xf64, #CSR>
}

// Addition with a dense operand can produce coordinates outside the sparse
// input and therefore cannot reuse its structure.
// CHECK-LABEL: func.func @structure_not_reused_for_union(
// CHECK-NOT:     sparse_tensor.assemble
// CHECK:         tensor.insert
func.func @structure_not_reused_for_union(
    %sparse: tensor<4x5xf64, #CSR>,
    %dense: tensor<4x5xf64>) -> tensor<4x5xf64, #CSR> {
  %empty = tensor.empty() : tensor<4x5xf64, #CSR>
  %result = linalg.generic #trait_scale
      ins(%sparse, %dense : tensor<4x5xf64, #CSR>, tensor<4x5xf64>)
      outs(%empty : tensor<4x5xf64, #CSR>) {
    ^bb0(%sparseValue: f64, %denseValue: f64, %unused: f64):
      %sum = arith.addf %sparseValue, %denseValue : f64
      linalg.yield %sum : f64
  } -> tensor<4x5xf64, #CSR>
  return %result : tensor<4x5xf64, #CSR>
}

// Equal sparse encodings do not prove that two independent tensors have the
// same stored coordinates. Do not apply the structure-preserving lowering
// when the destination is not derived from the sample.
// CHECK-LABEL: func.func @output_structure_not_proven(
// CHECK-NOT:     sparse_tensor.assemble
// CHECK:         linalg.generic
// CHECK:         return
func.func @output_structure_not_proven(
    %sample: tensor<4x5xf64, #CSR>,
    %lhs: tensor<4x3xf64>,
    %rhs: tensor<3x5xf64>,
    %output: tensor<4x5xf64, #CSR>) -> tensor<4x5xf64, #CSR> {
  %result = linalg.generic #trait_value_weighted_sddmm
      ins(%sample, %lhs, %rhs : tensor<4x5xf64, #CSR>,
                                  tensor<4x3xf64>, tensor<3x5xf64>)
      outs(%output : tensor<4x5xf64, #CSR>) {
    ^bb0(%s: f64, %a: f64, %b: f64, %acc: f64):
      %ab = arith.mulf %a, %b : f64
      %weighted = arith.mulf %s, %ab : f64
      %next = arith.addf %acc, %weighted : f64
      linalg.yield %next : f64
  } -> tensor<4x5xf64, #CSR>
  return %result : tensor<4x5xf64, #CSR>
}
