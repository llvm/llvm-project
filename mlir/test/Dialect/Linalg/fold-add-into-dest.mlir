// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!type = tensor<2048x2048xf32>
func.func @fold_add_on_two_matmuls(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = tensor.empty() : !type
  %5 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %6 = linalg.matmul ins(%arg1, %0 : !type, !type) outs(%5 : !type) -> !type
  %7 = linalg.add ins(%3, %6 : !type, !type) outs(%1 : !type) -> !type
  return %7 : !type
}

// CHECK-LABEL: func.func @fold_add_on_two_matmuls(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}})
// CHECK-NEXT: %[[DENSE:.*]] = arith.constant dense<1.11
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00
// CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
// CHECK-NEXT: %[[FILLED:.*]] = linalg.fill ins(%[[ZERO]] : {{.*}}) outs(%[[EMPTY]] : {{.*}})
// CHECK-NEXT: %[[ACC:.+]] = linalg.matmul ins(%[[ARG0]], %[[DENSE]] : {{.*}}) outs(%[[FILLED]] : {{.*}})
// CHECK-NEXT: %[[RES:.+]] = linalg.matmul ins(%[[ARG1]], %[[DENSE]] : {{.*}}) outs(%[[ACC]] : {{.*}})
// CHECK-NOT: linalg.add
// CHECK-NEXT: return %[[RES]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}
// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_of_add_as_orig_dest_not_additive_zero(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.matmul ins(%arg1, %0 : !type, !type) outs(%0 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_of_add_as_orig_dest_not_additive_zero
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_of_add_as_contraction_result_has_multiple_users(%arg0: !type, %arg1: !type) -> (!type, !type) {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.matmul ins(%arg1, %0 : !type, !type) outs(%0 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  %6 = linalg.mul ins(%4, %arg0 : !type, !type) outs(%1 : !type) -> !type
  return %5, %6 : !type, !type
}

// CHECK-LABEL: func.func @expect_no_fold_of_add_as_contraction_result_has_multiple_users
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.add
// CHECK-NEXT: linalg.mul
// CHECK-NEXT: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

!type = tensor<2048x2048xf32>
func.func @fold_add_on_matmul_and_func_arg(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %5 = linalg.add ins(%3, %arg1 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @fold_add_on_matmul_and_func_arg
// CHECK: %[[RES:.+]] = linalg.matmul
// CHECK-NOT: linalg.add
// CHECK-NEXT: return %[[RES]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_of_add_as_operands_do_not_dominate_each_other(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.add ins(%3, %3 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_of_add_as_operands_do_not_dominate_each_other
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_of_add_as_dominated_op_is_not_a_contraction(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.sub ins(%arg1, %0 : !type, !type) outs(%2 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_of_add_as_dominated_op_is_not_a_contraction
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.sub
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>  // NB: not an ordered projection

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_of_add_as_dest_accumulation_is_not_identity_mapped(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.generic { indexing_maps = [#map0, #map1, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"] }
    ins(%arg0, %0: !type, !type) outs(%2: !type) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %5 = arith.mulf %a, %b : f32
        %6 = arith.addf %c, %5 : f32
        linalg.yield %6 : f32
  } -> !type
  %4 = linalg.add ins(%3, %arg1 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_of_add_as_dest_accumulation_is_not_identity_mapped
// CHECK: linalg.fill
// CHECK-NEXT: linalg.generic
// CHECK: linalg.add
// CHECK-NEXT: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>  // NB: is an ordered projection

!type = tensor<2048x2048xf32>
func.func @fold_add_on_a_generic_and_an_argument(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.generic { indexing_maps = [#map0, #map1, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"] }
    ins(%arg0, %0: !type, !type) outs(%2: !type) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %5 = arith.mulf %a, %b : f32
        %6 = arith.addf %c, %5 : f32
        linalg.yield %6 : f32
  } -> !type
  %4 = linalg.add ins(%3, %arg1 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}

// CHECK-LABEL: func.func @fold_add_on_a_generic_and_an_argument
// CHECK: linalg.generic
// CHECK-NOT: linalg.add
// CHECK: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}

// -----

memref.global "private" constant @big_const : memref<2048x2048xf32> = dense<1.11111104> {alignment = 64 : i64}
func.func @expect_no_fold_due_to_no_memref_support(%arg0: memref<2048x2048xf32>, %arg1: memref<2048x2048xf32>) -> memref<2048x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @big_const  : memref<2048x2048xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048x2048xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2048x2048xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<2048x2048xf32>)
  linalg.matmul ins(%arg0, %0 : memref<2048x2048xf32>, memref<2048x2048xf32>) outs(%alloc_0 : memref<2048x2048xf32>)
  linalg.fill ins(%cst : f32) outs(%alloc : memref<2048x2048xf32>)
  linalg.matmul ins(%arg1, %0 : memref<2048x2048xf32>, memref<2048x2048xf32>) outs(%alloc : memref<2048x2048xf32>)
  linalg.add ins(%alloc_0, %alloc : memref<2048x2048xf32>, memref<2048x2048xf32>) outs(%alloc : memref<2048x2048xf32>)
  memref.dealloc %alloc_0 : memref<2048x2048xf32>
  return %alloc : memref<2048x2048xf32>
}

// CHECK-LABEL: func.func @expect_no_fold_due_to_no_memref_support
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: linalg.add
// CHECK: return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_add_into_dest
    } : !transform.any_op
    transform.yield
  }
}
