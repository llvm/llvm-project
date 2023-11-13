// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

#matvec_accesses = [
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}
#matvecmax_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"],
  kind = #vector.kind<maxf>
}

#mattransvec_accesses = [
  affine_map<(m, k) -> (k, m)>,
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m)>
]
#mattransvec_trait = {
  indexing_maps = #mattransvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#vecmat_accesses = [
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (m)>
]
#vecmat_trait = {
  indexing_maps = #vecmat_accesses,
  iterator_types = ["parallel", "reduction"]
}

#vecmattrans_accesses = [
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (k, m)>,
  affine_map<(m, k) -> (m)>
]
#vecmattrans_trait = {
  indexing_maps = #vecmattrans_accesses,
  iterator_types = ["parallel", "reduction"]
}

#redpar_vecmattrans_accesses = [
  affine_map<(m, k) -> (m)>,
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (k)>
]
#redpar_vecmattrans_trait = {
  indexing_maps = #redpar_vecmattrans_accesses,
  iterator_types = ["reduction", "parallel"]
}

// CHECK-LABEL: func @matvec_mk_k_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[B:.*1]]: vector<2xf32>
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[C]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_mk_k_m(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @matvec_mk_k_m_max
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[B:.*1]]: vector<2xf32>
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[C]] {kind = #vector.kind<maxf>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<maxf>} : vector<2xf32>, f32
func.func @matvec_mk_k_m_max(%A: vector<2x2xf32>,
                             %x: vector<2xf32>,
                             %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvecmax_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @matvec_km_k_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[B:.*1]]: vector<2xf32>
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[C]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_km_k_m(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #mattransvec_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @matvec_k_mk_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[B:.*1]]: vector<2xf32>
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[C]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_k_mk_m(%A: vector<2x2xf32>, 
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #vecmat_trait %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @matvec_k_km_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[B:.*1]]: vector<2xf32>
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[C]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_k_km_m(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #vecmattrans_trait %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @matvec_m_mk_k
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[B:.*1]]: vector<2xf32>
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[B]][0] : f32 from vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[C]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[B]][1] : f32 from vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_m_mk_k(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #redpar_vecmattrans_trait %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.op<"func.func">
    transform.yield
  }
}
