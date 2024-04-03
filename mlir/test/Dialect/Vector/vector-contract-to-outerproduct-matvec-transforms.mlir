// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

/// Tests for `vector.contract` -> `vector.outerproduct` transformations for
/// Matvec operations:
///   b += A * x.
/// (b and x are 1-d vectors, A is a 2-d matrix). ATM three different variants
/// are tested:
///   * plain (no mask, fixed-wdith vectors),
///   * masked (fixed-width vectors,
///   * scalable (mask + scalable vectors).
///
/// TODO: These tests were extracted from 2 different files. If you find the
/// formatting inconsistent, please update accordingly.

#matvec_accesses_1 = [
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m)>
]
#matvec_trait_1 = {
  indexing_maps = #matvec_accesses_1,
  iterator_types = ["parallel", "reduction"]
}

#matvecmax_trait = {
  indexing_maps = #matvec_accesses_1,
  iterator_types = ["parallel", "reduction"],
  kind = #vector.kind<maxnumf>
}

#matvec_accesses_2 = [
  affine_map<(m, k) -> (k, m)>,
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m)>
]
#matvec_trait_2 = {
  indexing_maps = #matvec_accesses_2,
  iterator_types = ["parallel", "reduction"]
}

#matvec_accesses_3 = [
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (m)>
]
#matvec_trait_3 = {
  indexing_maps = #matvec_accesses_3,
  iterator_types = ["parallel", "reduction"]
}

#matvec_accesses_4 = [
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (k, m)>,
  affine_map<(m, k) -> (m)>
]
#matvec_trait_4 = {
  indexing_maps = #matvec_accesses_4,
  iterator_types = ["parallel", "reduction"]
}

#matvec_accesses_5 = [
  affine_map<(k, m) -> (m, k)>,
  affine_map<(k, m) -> (k)>,
  affine_map<(k, m) -> (m)>
]
#matvec_trait_5 = {
  indexing_maps = #matvec_accesses_5,
  iterator_types = ["reduction", "parallel"]
}

#matvec_accesses_6 = [
  affine_map<(k, m) -> (k, m)>,
  affine_map<(k, m) -> (k)>,
  affine_map<(k, m) -> (m)>
]
#matvec_trait_6 = {
  indexing_maps = #matvec_accesses_6,
  iterator_types = ["reduction", "parallel"]
}

#matvec_accesses_7 = [
  affine_map<(k, m) -> (k)>,
  affine_map<(k, m) -> (m, k)>,
  affine_map<(k, m) -> (m)>
]
#matvec_trait_7 = {
  indexing_maps = #matvec_accesses_7,
  iterator_types = ["reduction", "parallel"]
}

#matvec_accesses_8 = [
  affine_map<(k, m) -> (k)>,
  affine_map<(k, m) -> (k, m)>,
  affine_map<(k, m) -> (m)>
]
#matvec_trait_8 = {
  indexing_maps = #matvec_accesses_8,
  iterator_types = ["reduction", "parallel"]
}

// ============================================================================
//  Matvec 1 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL: func @matvec_mk_k_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[X:.*1]]: vector<2xf32>
// CHECK-SAME: %[[B:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_mk_k_m(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_1 %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_matvec_mk_k_m(
// CHECK-SAME:      %{{.*}}: vector<2x3xf32>,
// CHECK-SAME:      %{{.*}}: vector<3xf32>,
// CHECK-SAME:      %{{.*}}: vector<2xf32>,
// CHECK-SAME:      %[[IN_MASK:.*]]: vector<2x3xi1>) -> vector<2xf32>
// CHECK:           %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [1, 0] : vector<2x3xi1> to vector<3x2xi1>
// CHECK:           %[[MASK0:.*]] = vector.extract %[[T_MASK]][0] : vector<2xi1> from vector<3x2xi1>
// CHECK:           vector.mask %[[MASK0]] { vector.outerproduct {{.*}} {kind = #vector.kind<add>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>

// CHECK:           %[[MASK1:.*]] = vector.extract %[[T_MASK]][1] : vector<2xi1> from vector<3x2xi1>
// CHECK:           vector.mask %[[MASK1]] { vector.outerproduct {{.*}} {kind = #vector.kind<add>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>

// CHECK:           %[[MASK2:.*]] = vector.extract %[[T_MASK]][2] : vector<2xi1> from vector<3x2xi1>
// CHECK:           vector.mask %[[MASK2]] { vector.outerproduct {{.*}} {kind = #vector.kind<add>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>
func.func @masked_matvec_mk_k_m(%A: vector<2x3xf32>,
                                %x: vector<3xf32>,
                                %b: vector<2xf32>,
                                %m: vector<2x3xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.contract #matvec_trait_1 %A, %x, %b
          : vector<2x3xf32>, vector<3xf32> into vector<2xf32> } : vector<2x3xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_matvec_mk_k_m_scalable_parallel_dim(
// CHECK-SAME:      %{{.*}}: vector<[2]x3xf32>,
// CHECK-SAME:      %{{.*}}: vector<3xf32>,
// CHECK-SAME:      %{{.*}}: vector<[2]xf32>,
// CHECK-SAME:      %[[IN_MASK:.*]]: vector<[2]x3xi1>) -> vector<[2]xf32>
// CHECK:           %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [1, 0] : vector<[2]x3xi1> to vector<3x[2]xi1>
// CHECK:           %[[MASK0:.*]] = vector.extract %[[T_MASK]][0] : vector<[2]xi1> from vector<3x[2]xi1>
// CHECK:           vector.mask %[[MASK0]] { vector.outerproduct {{.*}} {kind = #vector.kind<add>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>

// CHECK:           %[[MASK1:.*]] = vector.extract %[[T_MASK]][1] : vector<[2]xi1> from vector<3x[2]xi1>
// CHECK:           vector.mask %[[MASK1]] { vector.outerproduct {{.*}} {kind = #vector.kind<add>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>

// CHECK:           %[[MASK2:.*]] = vector.extract %[[T_MASK]][2] : vector<[2]xi1> from vector<3x[2]xi1>
// CHECK:           vector.mask %[[MASK2]] { vector.outerproduct {{.*}} {kind = #vector.kind<add>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>
func.func @masked_matvec_mk_k_m_scalable_parallel_dim(%A: vector<[2]x3xf32>,
                                                      %x: vector<3xf32>,
                                                      %b: vector<[2]xf32>,
                                                      %m: vector<[2]x3xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.contract #matvec_trait_1 %A, %x, %b
          : vector<[2]x3xf32>, vector<3xf32> into vector<[2]xf32> } : vector<[2]x3xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// ============================================================================
//  Matvec 1  - max (plain)
// ============================================================================
// CHECK-LABEL: func @matvec_mk_k_m_max
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[X:.*1]]: vector<2xf32>
// CHECK-SAME: %[[B:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[B]] {kind = #vector.kind<maxnumf>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<maxnumf>} : vector<2xf32>, f32
func.func @matvec_mk_k_m_max(%A: vector<2x2xf32>,
                             %x: vector<2xf32>,
                             %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvecmax_trait %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_matvec_mk_k_m_max(
// CHECK-SAME:      %{{.*}}: vector<2x3xf32>,
// CHECK-SAME:      %{{.*}}: vector<3xf32>,
// CHECK-SAME:      %{{.*}}: vector<2xf32>,
// CHECK-SAME:      %[[IN_MASK:.*]]: vector<2x3xi1>) -> vector<2xf32>
// CHECK:           %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [1, 0] : vector<2x3xi1> to vector<3x2xi1>
// CHECK:           %[[MASK0:.*]] = vector.extract %[[T_MASK]][0] : vector<2xi1> from vector<3x2xi1>
// CHECK:           vector.mask %[[MASK0]] { vector.outerproduct {{.*}} {kind = #vector.kind<maxnumf>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>

// CHECK:           %[[MASK1:.*]] = vector.extract %[[T_MASK]][1] : vector<2xi1> from vector<3x2xi1>
// CHECK:           vector.mask %[[MASK1]] { vector.outerproduct {{.*}} {kind = #vector.kind<maxnumf>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>

// CHECK:           %[[MASK2:.*]] = vector.extract %[[T_MASK]][2] : vector<2xi1> from vector<3x2xi1>
// CHECK:           vector.mask %[[MASK2]] { vector.outerproduct {{.*}} {kind = #vector.kind<maxnumf>} : vector<2xf32>, f32 } : vector<2xi1> -> vector<2xf32>
func.func @masked_matvec_mk_k_m_max(%A: vector<2x3xf32>,
                                    %x: vector<3xf32>,
                                    %b: vector<2xf32>,
                                    %m: vector<2x3xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.contract #matvecmax_trait %A, %x, %b
          : vector<2x3xf32>, vector<3xf32> into vector<2xf32> } : vector<2x3xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL:   func.func @masked_matvec_mk_k_m_max_scalable_parallel_dim(
// CHECK-SAME:      %{{.*}}: vector<[2]x3xf32>,
// CHECK-SAME:      %{{.*}}: vector<3xf32>,
// CHECK-SAME:      %{{.*}}: vector<[2]xf32>,
// CHECK-SAME:      %[[IN_MASK:.*]]: vector<[2]x3xi1>) -> vector<[2]xf32>
// CHECK:           %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [1, 0] : vector<[2]x3xi1> to vector<3x[2]xi1>
// CHECK:           %[[MASK0:.*]] = vector.extract %[[T_MASK]][0] : vector<[2]xi1> from vector<3x[2]xi1>
// CHECK:           vector.mask %[[MASK0]] { vector.outerproduct {{.*}} {kind = #vector.kind<maxnumf>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>

// CHECK:           %[[MASK1:.*]] = vector.extract %[[T_MASK]][1] : vector<[2]xi1> from vector<3x[2]xi1>
// CHECK:           vector.mask %[[MASK1]] { vector.outerproduct {{.*}} {kind = #vector.kind<maxnumf>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>

// CHECK:           %[[MASK2:.*]] = vector.extract %[[T_MASK]][2] : vector<[2]xi1> from vector<3x[2]xi1>
// CHECK:           vector.mask %[[MASK2]] { vector.outerproduct {{.*}} {kind = #vector.kind<maxnumf>} : vector<[2]xf32>, f32 } : vector<[2]xi1> -> vector<[2]xf32>
func.func @masked_matvec_mk_k_m_max_scalable_parallel_dim(%A: vector<[2]x3xf32>,
                                                          %x: vector<3xf32>,
                                                          %b: vector<[2]xf32>,
                                                          %m: vector<[2]x3xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.contract #matvecmax_trait %A, %x, %b
          : vector<[2]x3xf32>, vector<3xf32> into vector<[2]xf32> } : vector<[2]x3xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// ============================================================================
//  Matvec 2 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL: func @matvec_km_k_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[X:.*1]]: vector<2xf32>
// CHECK-SAME: %[[B:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_km_k_m(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_2 %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_matvec_km_k_m
// CHECK-SAME:  %[[A:.+]]: vector<2x4xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<4x2xi1>
func.func @masked_matvec_km_k_m(%A: vector<2x4xf32>,
                                %x: vector<2xf32>,
                                %b: vector<4xf32>, 
                                %mask: vector<4x2xi1>) -> vector<4xf32> {
  // CHECK:         vector.transpose %[[MASK]]
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_2 %A, %x, %b
      : vector<2x4xf32>, vector<2xf32>, vector<4xf32> into vector<4xf32>
  } : vector<4x2xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @masked_matvec_km_k_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<2x[4]xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<[4]x2xi1>
func.func @masked_matvec_km_k_m_scalable_parallel_dim(%A: vector<2x[4]xf32>,
                                                      %x: vector<2xf32>,
                                                      %b: vector<[4]xf32>,
                                                      %mask: vector<[4]x2xi1>) -> vector<[4]xf32> {
  // CHECK:         vector.transpose %[[MASK]]
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_2 %A, %x, %b
      : vector<2x[4]xf32>, vector<2xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<[4]x2xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// ============================================================================
//  Matvec 3 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL: func @matvec_k_mk_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[X:.*1]]: vector<2xf32>
// CHECK-SAME: %[[B:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[T3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T5:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK: %[[T6:.*]] = vector.outerproduct %[[T4]], %[[T5]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T7:.*]] = vector.extract %[[T3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK: %[[T9:.*]] = vector.outerproduct %[[T7]], %[[T8]], %[[T6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_k_mk_m(%A: vector<2x2xf32>, 
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_3 %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_matvec_k_mk_m
// CHECK-SAME:  %[[A:.+]]: vector<4x2xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<4x2xi1>
func.func @masked_matvec_k_mk_m(%A: vector<4x2xf32>,
                                %x: vector<2xf32>,
                                %b: vector<4xf32>,
                                %mask: vector<4x2xi1>) -> vector<4xf32> {
  // CHECK:         vector.transpose %[[A]]
  // CHECK:         vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
      vector.contract #matvec_trait_3 %x, %A, %b
        : vector<2xf32>, vector<4x2xf32>, vector<4xf32> into vector<4xf32>
  } : vector<4x2xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @masked_matvec_k_mk_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<[4]x2xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<[4]x2xi1>
func.func @masked_matvec_k_mk_m_scalable_parallel_dim(%A: vector<[4]x2xf32>,
                                                      %x: vector<2xf32>,
                                                      %b: vector<[4]xf32>,
                                                      %mask: vector<[4]x2xi1>) -> vector<[4]xf32> {
  // CHECK:         vector.transpose %[[A]]
  // CHECK:         vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
      vector.contract #matvec_trait_3 %x, %A, %b
        : vector<2xf32>, vector<[4]x2xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<[4]x2xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// ============================================================================
//  Matvec 4 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL: func @matvec_k_km_m
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[X:.*1]]: vector<2xf32>
// CHECK-SAME: %[[B:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @matvec_k_km_m(%A: vector<2x2xf32>,
                         %x: vector<2xf32>,
                         %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_4 %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_matvec_k_km_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<2x[4]xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<[4]x2xi1>
func.func @masked_matvec_k_km_m_scalable_parallel_dim(%A: vector<2x[4]xf32>,
                                                      %x: vector<2xf32>,
                                                      %b: vector<[4]xf32>,
                                                      %mask: vector<[4]x2xi1>) -> vector<[4]xf32> {
  // CHECK:         vector.transpose %[[MASK]]
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_4 %x, %A, %b
      : vector<2xf32>, vector<2x[4]xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<[4]x2xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// CHECK-LABEL: @masked_matvec_k_km_m
// CHECK-SAME:  %[[A:.+]]: vector<2x4xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<4x2xi1>
func.func @masked_matvec_k_km_m(%A: vector<2x4xf32>,
                                %x: vector<2xf32>,
                                %b: vector<4xf32>,
                                %mask: vector<4x2xi1>) -> vector<4xf32> {
  // CHECK:         vector.transpose %[[MASK]]
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_4 %x, %A, %b
      : vector<2xf32>, vector<2x4xf32>, vector<4xf32> into vector<4xf32>
  } : vector<4x2xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// ============================================================================
//  Matvec 5 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL:   func.func @tmatvec_mk_k_m(
// CHECK-SAME:      %[[A:.*]]: vector<2x2xf32>,
// CHECK-SAME:      %[[X:.*]]: vector<2xf32>,
// CHECK-SAME:      %[[B:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK:           %[[VAL_3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK:           %[[VAL_4:.*]] = vector.extract %[[VAL_3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:           %[[VAL_5:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK:           %[[VAL_6:.*]] = vector.outerproduct %[[VAL_4]], %[[VAL_5]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK:           %[[VAL_7:.*]] = vector.extract %[[VAL_3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:           %[[VAL_8:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = vector.outerproduct %[[VAL_7]], %[[VAL_8]], %[[VAL_6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @tmatvec_mk_k_m(%A: vector<2x2xf32>,
                          %x: vector<2xf32>,
                          %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_5 %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_tmatvec_mk_k_m
// CHECK-SAME:  %[[A:.+]]: vector<4x2xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x4xi1>
func.func @masked_tmatvec_mk_k_m(%A: vector<4x2xf32>,
                                 %x: vector<2xf32>,
                                 %b: vector<4xf32>,
                                 %mask: vector<2x4xi1>) -> vector<4xf32> {
  // CHECK:         vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_5 %A, %x, %b
      : vector<4x2xf32>, vector<2xf32>, vector<4xf32> into vector<4xf32>
  } : vector<2x4xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @masked_tmatvec_mk_k_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<[4]x2xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x[4]xi1>
func.func @masked_tmatvec_mk_k_m_scalable_parallel_dim(%A: vector<[4]x2xf32>,
                                                       %x: vector<2xf32>,
                                                       %b: vector<[4]xf32>,
                                                       %mask: vector<2x[4]xi1>) -> vector<[4]xf32> {
  // CHECK:         vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_5 %A, %x, %b
      : vector<[4]x2xf32>, vector<2xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<2x[4]xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// ============================================================================
//  Matvec 6 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL:   func.func @tmatvec_km_k_m(
// CHECK-SAME:      %[[A:.*]]: vector<2x2xf32>,
// CHECK-SAME:      %[[X:.*]]: vector<2xf32>,
// CHECK-SAME:      %[[B:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK:           %[[VAL_3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:           %[[VAL_4:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK:           %[[VAL_5:.*]] = vector.outerproduct %[[VAL_3]], %[[VAL_4]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK:           %[[VAL_6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:           %[[VAL_7:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK:           %[[VAL_8:.*]] = vector.outerproduct %[[VAL_6]], %[[VAL_7]], %[[VAL_5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @tmatvec_km_k_m(%A: vector<2x2xf32>,
                          %x: vector<2xf32>,
                          %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_6 %A, %x, %b : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_tmatvec_km_k_m
// CHECK-SAME:  %[[A:.+]]: vector<2x4xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x4xi1>
func.func @masked_tmatvec_km_k_m(%A: vector<2x4xf32>,
                                 %x: vector<2xf32>,
                                 %b: vector<4xf32>,
                                 %mask: vector<2x4xi1>) -> vector<4xf32> {
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_6 %A, %x, %b
      : vector<2x4xf32>, vector<2xf32>, vector<4xf32> into vector<4xf32>
  } : vector<2x4xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @masked_tmatvec_km_k_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<2x[4]xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x[4]xi1>
func.func @masked_tmatvec_km_k_m_scalable_parallel_dim(%A: vector<2x[4]xf32>,
                                                       %x: vector<2xf32>,
                                                       %b: vector<[4]xf32>,
                                                       %mask: vector<2x[4]xi1>) -> vector<[4]xf32> {
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_6 %A, %x, %b
      : vector<2x[4]xf32>, vector<2xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<2x[4]xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// ============================================================================
//  Matvec 7 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL:   func.func @tmatvec_k_mk_m(
// CHECK-SAME:      %[[A:.*]]: vector<2x2xf32>,
// CHECK-SAME:      %[[X:.*]]: vector<2xf32>,
// CHECK-SAME:      %[[B:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK:           %[[VAL_3:.*]] = vector.transpose %[[A]], [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK:           %[[VAL_4:.*]] = vector.extract %[[VAL_3]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:           %[[VAL_5:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK:           %[[VAL_6:.*]] = vector.outerproduct %[[VAL_4]], %[[VAL_5]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK:           %[[VAL_7:.*]] = vector.extract %[[VAL_3]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:           %[[VAL_8:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK:           %[[VAL_9:.*]] = vector.outerproduct %[[VAL_7]], %[[VAL_8]], %[[VAL_6]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @tmatvec_k_mk_m(%A: vector<2x2xf32>,
                          %x: vector<2xf32>,
                          %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_7 %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_tmatvec_k_mk_m
// CHECK-SAME:  %[[A:.+]]: vector<4x2xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x4xi1>
func.func @masked_tmatvec_k_mk_m(%A: vector<4x2xf32>,
                                 %x: vector<2xf32>,
                                 %b: vector<4xf32>,
                                 %mask: vector<2x4xi1>) -> vector<4xf32> {
  // CHECK:         vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_7 %x, %A, %b
      : vector<2xf32>, vector<4x2xf32>, vector<4xf32> into vector<4xf32>
  } : vector<2x4xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @masked_tmatvec_k_mk_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<[4]x2xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x[4]xi1>
func.func @masked_tmatvec_k_mk_m_scalable_parallel_dim(%A: vector<[4]x2xf32>,
                                                       %x: vector<2xf32>,
                                                       %b: vector<[4]xf32>,
                                                       %mask: vector<2x[4]xi1>) -> vector<[4]xf32> {
  // CHECK:         vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_7 %x, %A, %b
      : vector<2xf32>, vector<[4]x2xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<2x[4]xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// ============================================================================
//  Matvec 8 (plain + masked + scalable)
// ============================================================================
// CHECK-LABEL: func @tmatvec_m_mk_k
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>
// CHECK-SAME: %[[X:.*1]]: vector<2xf32>
// CHECK-SAME: %[[B:.*2]]: vector<2xf32>
// CHECK: %[[T3:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[X]][0] : f32 from vector<2xf32>
// CHECK: %[[T5:.*]] = vector.outerproduct %[[T3]], %[[T4]], %[[B]] {kind = #vector.kind<add>} : vector<2xf32>, f32
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[T7:.*]] = vector.extract %[[X]][1] : f32 from vector<2xf32>
// CHECK: %[[T8:.*]] = vector.outerproduct %[[T6]], %[[T7]], %[[T5]] {kind = #vector.kind<add>} : vector<2xf32>, f32
func.func @tmatvec_m_mk_k(%A: vector<2x2xf32>,
                          %x: vector<2xf32>,
                          %b: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait_8 %x, %A, %b : vector<2xf32>, vector<2x2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @masked_tmatvec_k_km_m
// CHECK-SAME:  %[[A:.+]]: vector<2x4xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<4xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x4xi1>
func.func @masked_tmatvec_k_km_m(%A: vector<2x4xf32>,
                                 %x: vector<2xf32>,
                                 %b: vector<4xf32>,
                                 %mask: vector<2x4xi1>) -> vector<4xf32> {
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<4xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_8 %x, %A, %b
      : vector<2xf32>, vector<2x4xf32>, vector<4xf32> into vector<4xf32>
  } : vector<2x4xi1> -> vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @masked_tmatvec_k_km_m_scalable_parallel_dim
// CHECK-SAME:  %[[A:.+]]: vector<2x[4]xf32>
// CHECK-SAME:  %[[X:.+]]: vector<2xf32>
// CHECK-SAME:  %[[B:.+]]: vector<[4]xf32>
// CHECK-SAME:  %[[MASK:.+]]: vector<2x[4]xi1>
func.func @masked_tmatvec_k_km_m_scalable_parallel_dim(%A: vector<2x[4]xf32>,
                                                       %x: vector<2xf32>,
                                                       %b: vector<[4]xf32>,
                                                       %mask: vector<2x[4]xi1>) -> vector<[4]xf32> {
  // CHECK-NOT:     vector.transpose %[[A]]
  // CHECK-NOT:     vector.transpose %[[MASK]]
  // CHECK-COUNT-2: vector.mask %{{.*}} { vector.outerproduct %{{.*}}, %{{.*}}, %{{.*}} {kind = #vector.kind<add>} : vector<[4]xf32>, f32 }
  %res = vector.mask %mask {
    vector.contract #matvec_trait_8 %x, %A, %b
      : vector<2xf32>, vector<2x[4]xf32>, vector<[4]xf32> into vector<[4]xf32>
  } : vector<2x[4]xi1> -> vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// Unrolling scalable reduction dim is not supported - bail out
// CHECK-LABEL: @masked_extract_contract2_scalable_reduction_dim(
// CHECK:         vector.contract {{.*}} : vector<[2]x[3]xf32>, vector<[3]xf32> into vector<[2]xf32>
func.func @masked_extract_contract2_scalable_reduction_dim(%arg0: vector<[2]x[3]xf32>,
                                    %arg1: vector<[3]xf32>,
                                    %arg2: vector<[2]xf32>,
                                    %m: vector<[2]x[3]xi1>) -> vector<[2]xf32> {
  %0 = vector.mask %m { vector.contract #matvec_trait_1 %arg0, %arg1, %arg2
          : vector<[2]x[3]xf32>, vector<[3]xf32> into vector<[2]xf32> } : vector<[2]x[3]xi1> -> vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// ============================================================================
//  TD sequence
// ============================================================================
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.op<"func.func">
    transform.yield
  }
}
