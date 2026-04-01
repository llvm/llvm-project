// RUN: mlir-opt -affine-simplify-with-bounds %s | FileCheck %s

// CHECK-LABEL: func @many_to_one_static_tail
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index
// CHECK-DAG:     %[[LIN:.*]] = affine.linearize_index disjoint [%[[B]], %[[C]]] by (8, 8)
// CHECK-DAG:     return %[[A]], %[[LIN]]
func.func @many_to_one_static_tail(%a: index, %b: index, %c: index) -> (index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (4, 8, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, 64) : index, index
  return %1#0, %1#1 : index, index
}

// -----

// CHECK-LABEL: func @many_to_one_dynamic_tail
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[DYN:.*]]: index
// CHECK-DAG:     %[[LIN:.*]] = affine.linearize_index disjoint [%[[B]], %[[C]]] by (%[[DYN]], 8)
// CHECK-DAG:     return %[[A]], %[[LIN]]
func.func @many_to_one_dynamic_tail(%a: index, %b: index, %c: index, %dyn: index) -> (index, index) {
  %dyn_times_8 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%dyn]
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (4, %dyn, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, %dyn_times_8) : index, index
  return %1#0, %1#1 : index, index
}

// -----

// CHECK-LABEL: func @one_to_many_static_tail
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index
// CHECK-DAG:     %[[DELIN:.*]]:2 = affine.delinearize_index %[[B]] into (8, 8)
// CHECK-DAG:     return %[[A]], %[[DELIN]]#0, %[[DELIN]]#1
func.func @one_to_many_static_tail(%a: index, %b: index) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b] by (4, 64) : index
  %1:3 = affine.delinearize_index %0 into (4, 8, 8) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// CHECK-LABEL: func @one_to_many_dynamic_tail
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[DYN:.*]]: index
// CHECK-DAG:     %[[DELIN:.*]]:2 = affine.delinearize_index %[[B]] into (%[[DYN]], 8)
// CHECK-DAG:     return %[[A]], %[[DELIN]]#0, %[[DELIN]]#1
func.func @one_to_many_dynamic_tail(%a: index, %b: index, %dyn: index) -> (index, index, index) {
  %dyn_times_8 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%dyn]
  %0 = affine.linearize_index disjoint [%a, %b] by (4, %dyn_times_8) : index
  %1:3 = affine.delinearize_index %0 into (4, %dyn, 8) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// CHECK-LABEL: func @one_to_one_dynamic_tail
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[DYN:.*]]: index
// CHECK-DAG:     return %[[A]], %[[B]]
func.func @one_to_one_dynamic_tail(%a: index, %b: index, %dyn: index) -> (index, index) {
  %dyn_ceildiv = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%dyn]
  %dyn_ceildiv_dup = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%dyn]
  %0 = affine.linearize_index disjoint [%a, %b] by (4, %dyn_ceildiv) : index
  %1:2 = affine.delinearize_index %0 into (4, %dyn_ceildiv_dup) : index, index
  return %1#0, %1#1 : index, index
}

// -----

// Mixed: many-to-one and one-to-one in the same pair.
// The many-to-one pattern should match [%c, %d] -> 1 delin dim,
// and canonicalization handles the 1:1 prefix.
// CHECK-LABEL: func @mixed_many_to_one_and_one_to_one
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[D:.*]]: index
// CHECK-DAG:     %[[LIN:.*]] = affine.linearize_index disjoint [%[[C]], %[[D]]] by (4, 8)
// CHECK-DAG:     return %[[A]], %[[B]], %[[LIN]]
func.func @mixed_many_to_one_and_one_to_one(%a: index, %b: index, %c: index, %d: index) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c, %d] by (2, 3, 4, 8) : index
  %1:3 = affine.delinearize_index %0 into (2, 3, 32) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// Partial match: only tail matches, prefix is left as residual.
// CHECK-LABEL: func @partial_tail_many_to_one
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[D:.*]]: index
// CHECK:         %[[RESLIN:.*]] = affine.linearize_index disjoint [%[[A]], %[[B]]] by (5, 3)
// CHECK:         %[[RESDELIN:.*]]:2 = affine.delinearize_index %[[RESLIN]] into (7, 9)
// CHECK:         %[[TAILLIN:.*]] = affine.linearize_index disjoint [%[[C]], %[[D]]] by (4, 8)
// CHECK:         return %[[RESDELIN]]#0, %[[RESDELIN]]#1, %[[TAILLIN]]
func.func @partial_tail_many_to_one(%a: index, %b: index, %c: index, %d: index) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c, %d] by (5, 3, 4, 8) : index
  %1:3 = affine.delinearize_index %0 into (7, 9, 32) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// Many-to-one with no outer bound: all basis elements consumed.
// The outermost delinearize result (unbounded) passes through from the
// outermost linearize input.
// CHECK-LABEL: func @many_to_one_no_outer_bound
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index
// CHECK-DAG:     %[[LIN:.*]] = affine.linearize_index disjoint [%[[B]], %[[C]]] by (8, 8)
// CHECK-DAG:     return %[[A]], %[[LIN]]
func.func @many_to_one_no_outer_bound(%a: index, %b: index, %c: index) -> (index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (8, 8) : index
  %1:2 = affine.delinearize_index %0 into (64) : index, index
  return %1#0, %1#1 : index, index
}

// -----

// One-to-many with no outer bound: all basis elements consumed.
// CHECK-LABEL: func @one_to_many_no_outer_bound
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index
// CHECK-DAG:     %[[DELIN:.*]]:2 = affine.delinearize_index %[[B]] into (8, 8)
// CHECK-DAG:     return %[[A]], %[[DELIN]]#0, %[[DELIN]]#1
func.func @one_to_many_no_outer_bound(%a: index, %b: index) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b] by (64) : index
  %1:3 = affine.delinearize_index %0 into (8, 8) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// Partial match with empty residual linearize.
// CHECK-LABEL: func @partial_match_empty_residual_lin
// CHECK-SAME:    %[[A:.*]]: index, %[[B:.*]]: index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         return %[[C0]], %[[A]], %[[B]]
func.func @partial_match_empty_residual_lin(%a: index, %b: index) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b] by (4, 8) : index
  %1:3 = affine.delinearize_index %0 into (10, 4, 8) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// Negative test: no disjoint flag.
// CHECK-LABEL: func @no_disjoint
// CHECK:         affine.linearize_index [
// CHECK:         affine.delinearize_index
func.func @no_disjoint(%a: index, %b: index, %c: index) -> (index, index) {
  %0 = affine.linearize_index [%a, %b, %c] by (4, 8, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, 64) : index, index
  return %1#0, %1#1 : index, index
}

// -----

// Negative test: products don't match.
// CHECK-LABEL: func @products_dont_match
// CHECK:         affine.linearize_index disjoint
// CHECK:         affine.delinearize_index
func.func @products_dont_match(%a: index, %b: index, %c: index) -> (index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (4, 8, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, 63) : index, index
  return %1#0, %1#1 : index, index
}

// -----

// Negative test: input not from linearize.
// CHECK-LABEL: func @input_not_linearize
// CHECK:         affine.delinearize_index %{{.*}} into
func.func @input_not_linearize(%x: index) -> (index, index) {
  %0:2 = affine.delinearize_index %x into (4, 8) : index, index
  return %0#0, %0#1 : index, index
}
