// RUN: mlir-opt %s -remove-dead-values="canonicalize=0" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -remove-dead-values="canonicalize=1" -split-input-file | FileCheck %s --check-prefix=CHECK-CANONICALIZE

// The IR is updated regardless of memref.global private constant
//
module {
  // CHECK: memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]> {alignment = 16 : i64}
  memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]> {alignment = 16 : i64}
  func.func @main(%arg0: i32) -> i32 {
    %0 = tensor.empty() : tensor<10xbf16>
    // CHECK-NOT: memref.get_global
    %1 = memref.get_global @__constant_4xi32 : memref<4xi32>
    // CHECK-NOT: tensor.empty
    return %arg0 : i32
  }
}

// -----

// Dead values are removed from the IR even if the module has a name
//
module @named_module_acceptable {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tensor.empty() : tensor<10xbf16>
    // CHECK-NOT: tensor.empty
    return %arg0 : tensor<10xf32>
  }
}

// -----

// The IR contains both conditional and unconditional branches with a loop
// in which the last cf.cond_br is referncing the first cf.br
//
func.func @acceptable_ir_has_cleanable_loop_of_conditional_and_branch_op(%arg0: i1) {
  %non_live = arith.constant 0 : i32
  // CHECK-NOT: arith.constant
  cf.br ^bb1(%non_live : i32)
  // CHECK: cf.br ^[[BB1:bb[0-9]+]]
^bb1(%non_live_1 : i32):
  // CHECK: ^[[BB1]]:
  %non_live_5 = arith.constant 1 : i32
  cf.br ^bb3(%non_live_1, %non_live_5 : i32, i32)
  // CHECK: cf.br ^[[BB3:bb[0-9]+]]
  // CHECK-NOT: i32
^bb3(%non_live_2 : i32, %non_live_6 : i32):
  // CHECK: ^[[BB3]]:
  cf.cond_br %arg0, ^bb1(%non_live_2 : i32), ^bb4(%non_live_2 : i32)
  // CHECK: cf.cond_br %arg0, ^[[BB1]], ^[[BB4:bb[0-9]+]]
^bb4(%non_live_4 : i32):
  // CHECK: ^[[BB4]]:
  return
}

// -----

// Checking that iter_args are properly handled
//
// CHECK-CANONICALIZE-LABEL: func @cleanable_loop_iter_args_value
func.func @cleanable_loop_iter_args_value(%arg0: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %non_live = arith.constant 0 : index
  // CHECK-CANONICALIZE: [[RESULT:%.+]] = scf.for [[ARG_1:%.*]] = %c0 to %c10 step %c1 iter_args([[ARG_2:%.*]] = %arg0) -> (index) {
  %result, %result_non_live = scf.for %i = %c0 to %c10 step %c1 iter_args(%live_arg = %arg0, %non_live_arg = %non_live) -> (index, index) {
    // CHECK-CANONICALIZE: [[SUM:%.+]] = arith.addi [[ARG_2]], [[ARG_1]] : index
    %new_live = arith.addi %live_arg, %i : index
    // CHECK-CANONICALIZE: scf.yield [[SUM:%.+]]
    scf.yield %new_live, %non_live_arg : index, index
  }
  // CHECK-CANONICALIZE: return [[RESULT]] : index
  return %result : index
}

// -----

// Checking that the arguments of linalg.generic are properly handled
// All code below is removed as a result of the pass
//
#map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // CHECK-LABEL: @dead_linalg_generic
  func.func @dead_linalg_generic() {
    %cst_3 = arith.constant dense<54> : tensor<1x25x13xi32>
    %cst_7 = arith.constant dense<11> : tensor<1x25x13xi32>
    // CHECK-NOT: arith.constant
    %0 = tensor.empty() : tensor<1x25x13xi32>
    // CHECK-NOT: tensor
    %1 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_3, %cst_7 : tensor<1x25x13xi32>, tensor<1x25x13xi32>) outs(%0 : tensor<1x25x13xi32>) {
    // CHECK-NOT: linalg.generic
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %29 = arith.xori %in, %in_15 : i32
      // CHECK-NOT: arith.xori
      linalg.yield %29 : i32
      // CHECK-NOT: linalg.yield
    } -> tensor<1x25x13xi32>
    return
  }
}

// -----

// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK-LABEL: func.func private @clean_func_op_remove_argument_and_return_value() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:       func.func @main(%[[arg0:.*]]: i32) {
// CHECK-NEXT:    call @clean_func_op_remove_argument_and_return_value() : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func private @clean_func_op_remove_argument_and_return_value(%arg0: i32) -> (i32) {
  return %arg0 : i32
}
func.func @main(%arg0 : i32) {
  %non_live = func.call @clean_func_op_remove_argument_and_return_value(%arg0) : (i32) -> (i32)
  return
}

// -----

// CHECK-LABEL: func.func private @clean_func_op_remove_side_effecting_op() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func private @clean_func_op_remove_side_effecting_op(%arg0: i32) -> (i32) {
  // vector.print has a side effect but the op is dead.
  vector.print %arg0 : i32
  return %arg0 : i32
}

// -----

// %arg0 is not live because it is never used. %arg1 is not live because its
// user `arith.addi` doesn't have any uses and the value that it is forwarded to
// (%non_live_0) also doesn't have any uses.
//
// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK-LABEL: func.func private @clean_func_op_remove_arguments() -> i32 {
// CHECK-NEXT:    %[[c0:.*]] = arith.constant 0
// CHECK-NEXT:    return %[[c0]]
// CHECK-NEXT:  }
// CHECK:       func.func @main(%[[arg2:.*]]: memref<i32>, %[[arg3:.*]]: i32, %[[DEVICE:.*]]: i32) -> (i32, memref<i32>) {
// CHECK-NEXT:    %[[live:.*]] = test.call_on_device @clean_func_op_remove_arguments(), %[[DEVICE]] : (i32) -> i32
// CHECK-NEXT:    return %[[live]], %[[arg2]]
// CHECK-NEXT:  }
func.func private @clean_func_op_remove_arguments(%arg0 : memref<i32>, %arg1 : i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %non_live = arith.addi %arg1, %arg1 : i32
  return %c0, %arg1 : i32, i32
}
func.func @main(%arg2 : memref<i32>, %arg3 : i32, %device : i32) -> (i32, memref<i32>) {
  %live, %non_live_0 = test.call_on_device @clean_func_op_remove_arguments(%arg2, %arg3), %device : (memref<i32>, i32, i32) -> (i32, i32)
  return %live, %arg2 : i32, memref<i32>
}

// -----

// Even though %non_live_0 is not live, the first return value of
// @clean_func_op_remove_return_values isn't removed because %live is live
// (liveness is checked across all callers).
//
// Also, the second return value of @clean_func_op_remove_return_values is
// removed despite %c0 being live because neither %non_live nor %non_live_1 were
// live (removal doesn't depend on the liveness of the operand itself but on the
// liveness of where it is forwarded).
//
// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK:       func.func private @clean_func_op_remove_return_values(%[[arg0:.*]]: memref<i32>) -> i32 {
// CHECK-NEXT:    %[[c0]] = arith.constant 0
// CHECK-NEXT:    memref.store %[[c0]], %[[arg0]][]
// CHECK-NEXT:    return %[[c0]]
// CHECK-NEXT:  }
// CHECK:       func.func @main(%[[arg1:.*]]: memref<i32>) -> i32 {
// CHECK-NEXT:    %[[live:.*]] = call @clean_func_op_remove_return_values(%[[arg1]]) : (memref<i32>) -> i32
// CHECK-NEXT:    %[[non_live_0:.*]] = call @clean_func_op_remove_return_values(%[[arg1]]) : (memref<i32>) -> i32
// CHECK-NEXT:    return %[[live]] : i32
// CHECK-NEXT:  }
func.func private @clean_func_op_remove_return_values(%arg0 : memref<i32>) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %arg0[] : memref<i32>
  return %c0, %c0 : i32, i32
}
func.func @main(%arg1 : memref<i32>) -> (i32) {
  %live, %non_live = func.call @clean_func_op_remove_return_values(%arg1) : (memref<i32>) -> (i32, i32)
  %non_live_0, %non_live_1 = func.call @clean_func_op_remove_return_values(%arg1) : (memref<i32>) -> (i32, i32)
  return %live : i32
}

// -----

// None of the return values of @clean_func_op_dont_remove_return_values can be
// removed because the first one is forwarded to a live value %live and the
// second one is forwarded to a live value %live_0.
//
// CHECK-LABEL: func.func private @clean_func_op_dont_remove_return_values() -> (i32, i32) {
// CHECK-NEXT:    %[[c0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    return %[[c0]], %[[c0]] : i32, i32
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @main() -> (i32, i32) {
// CHECK-NEXT:    %[[live_and_non_live:.*]]:2 = call @clean_func_op_dont_remove_return_values() : () -> (i32, i32)
// CHECK-NEXT:    %[[non_live_0_and_live_0:.*]]:2 = call @clean_func_op_dont_remove_return_values() : () -> (i32, i32)
// CHECK-NEXT:    return %[[live_and_non_live]]#0, %[[non_live_0_and_live_0]]#1 : i32, i32
// CHECK-NEXT:  }
func.func private @clean_func_op_dont_remove_return_values() -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  return %c0, %c0 : i32, i32
}
func.func @main() -> (i32, i32) {
  %live, %non_live = func.call @clean_func_op_dont_remove_return_values() : () -> (i32, i32)
  %non_live_0, %live_0 = func.call @clean_func_op_dont_remove_return_values() : () -> (i32, i32)
  return %live, %live_0 : i32, i32
}

// -----

// Values kept:
//  (1) %non_live is not live. Yet, it is kept because %arg4 in `scf.condition`
//  forwards to it, which has to be kept. %arg4 in `scf.condition` has to be
//  kept because it forwards to %arg6 which is live.
//
//  (2) %arg5 is not live. Yet, it is kept because %live_0 forwards to it, which
//  also forwards to %live, which is live.
//
// Values not kept:
//  (1) %arg1 is not kept as an operand of `scf.while` because it only forwards
//  to %arg3, which is not kept. %arg3 is not kept because %arg3 is not live and
//  only %arg1 and %arg7 forward to it, such that neither of them forward
//  anywhere else. Thus, %arg7 is also not kept in the `scf.yield` op.
//
// Note that this cleanup cannot be done by the `canonicalize` pass.

// CHECK-LABEL: func.func @clean_region_branch_op_dont_remove_first_2_results_but_remove_first_operand(
// CHECK-SAME:      %[[arg0:.*]]: i1, %[[arg1:.*]]: i32, %[[arg2:.*]]: i32) -> i32 {
// CHECK-NEXT:    %[[p0:.*]] = ub.poison : i32
// CHECK-NEXT:    %[[while:.*]]:3 = scf.while (%{{.*}} = %[[p0]], %[[arg4:.*]] = %[[arg2]]) : (i32, i32) -> (i32, i32, i32) {
// CHECK-NEXT:      %[[add1:.*]] = arith.addi %[[arg4]], %[[arg4]] : i32
// CHECK-NEXT:      %[[p1:.*]] = ub.poison : i32
// CHECK-NEXT:      scf.condition(%[[arg0]]) %[[add1]], %[[arg4]], %[[p1]] : i32, i32, i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%{{.*}}: i32, %[[arg6:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:      %[[add2:.*]] = arith.addi %[[arg6]], %[[arg6]] : i32
// CHECK-NEXT:      %[[p2:.*]] = ub.poison : i32
// CHECK-NEXT:      scf.yield %[[p2]], %[[add2]] : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[while]]#0 : i32
// CHECK-NEXT:  }

// CHECK-CANONICALIZE:       func.func @clean_region_branch_op_dont_remove_first_2_results_but_remove_first_operand(%[[arg0:.*]]: i1, %[[arg1:.*]]: i32, %[[arg2:.*]]: i32) -> i32 {
// CHECK-CANONICALIZE:         %[[live_and_non_live:.*]]:2 = scf.while (%[[arg4:.*]] = %[[arg2]]) : (i32) -> (i32, i32) {
// CHECK-CANONICALIZE-NEXT:      %[[live_0:.*]] = arith.addi %[[arg4]], %[[arg4]]
// CHECK-CANONICALIZE:           scf.condition(%arg0) %[[live_0]], %[[arg4]] : i32, i32
// CHECK-CANONICALIZE-NEXT:    } do {
// CHECK-CANONICALIZE-NEXT:    ^bb0(%[[arg5:.*]]: i32, %[[arg6:.*]]: i32):
// CHECK-CANONICALIZE-NEXT:      %[[live_1:.*]] = arith.addi %[[arg6]], %[[arg6]]
// CHECK-CANONICALIZE:           scf.yield %[[live_1]] : i32
// CHECK-CANONICALIZE-NEXT:    }
// CHECK-CANONICALIZE-NEXT:    return %[[live_and_non_live]]#0
// CHECK-CANONICALIZE-NEXT:  }
func.func @clean_region_branch_op_dont_remove_first_2_results_but_remove_first_operand(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32) {
  %live, %non_live, %non_live_0 = scf.while (%arg3 = %arg1, %arg4 = %arg2) : (i32, i32) -> (i32, i32, i32) {
    %live_0 = arith.addi %arg4, %arg4 : i32
    %non_live_1 = arith.addi %arg3, %arg3 : i32
    scf.condition(%arg0) %live_0, %arg4, %non_live_1 : i32, i32, i32
  } do {
  ^bb0(%arg5: i32, %arg6: i32, %arg7: i32):
    %live_1 = arith.addi %arg6, %arg6 : i32
    scf.yield %arg7, %live_1 : i32, i32
  }
  return %live : i32
}

// -----

// Values kept:
//  (1) %live is kept because it is live.
//
//  (2) %non_live is not live. Yet, it is kept because %arg3 in `scf.condition`
//  forwards to it and this %arg3 has to be kept. This %arg3 in `scf.condition`
//  has to be kept because it forwards to %arg6, which forwards to %arg4, which
//  forwards to %live, which is live.
//
// Values not kept:
//  (1) %non_live_0 is not kept because %non_live_2 in `scf.condition` forwards
//  to it, which forwards to only %non_live_0 and %arg7, where both these are
//  not live and have no other value forwarding to them.
//
//  (2) %non_live_1 is not kept because %non_live_3 in `scf.condition` forwards
//  to it, which forwards to only %non_live_1 and %arg8, where both these are
//  not live and have no other value forwarding to them.
//
//  (3) %c2 is not kept because it only forwards to %arg10, which is not kept.
//
//  (4) %arg10 is not kept because only %c2 and %non_live_4 forward to it, none
//  of them forward anywhere else, and %arg10 is not.
//
//  (5) %arg7 and %arg8 are not kept because they are not live, %non_live_2 and
//  %non_live_3 forward to them, and both only otherwise forward to %non_live_0
//  and %non_live_1 which are not live and have no other predecessors.
//
// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK-CANONICALIZE:       func.func @clean_region_branch_op_remove_last_2_results_last_2_arguments_and_last_operand(%[[arg2:.*]]: i1) -> i32 {
// CHECK-CANONICALIZE-NEXT:    %[[c0:.*]] = arith.constant 0
// CHECK-CANONICALIZE-NEXT:    %[[c1:.*]] = arith.constant 1
// CHECK-CANONICALIZE:         %[[live_and_non_live:.*]]:2 = scf.while (%[[arg3:.*]] = %[[c0]], %[[arg4:.*]] = %[[c1]]) : (i32, i32) -> (i32, i32) {
// CHECK-CANONICALIZE-NEXT:      func.call @identity() : () -> ()
// CHECK-CANONICALIZE-NEXT:      scf.condition(%[[arg2]]) %[[arg3]], %[[arg4]] : i32, i32
// CHECK-CANONICALIZE-NEXT:    } do {
// CHECK-CANONICALIZE-NEXT:    ^bb0(%[[arg5:.*]]: i32, %[[arg6:.*]]: i32):
// CHECK-CANONICALIZE-NEXT:      scf.yield %[[arg6]], %[[arg5]] : i32, i32
// CHECK-CANONICALIZE-NEXT:    }
// CHECK-CANONICALIZE-NEXT:    return %[[live_and_non_live]]#1 : i32
// CHECK-CANONICALIZE-NEXT:  }
// CHECK-CANONICALIZE:       func.func private @identity() {
// CHECK-CANONICALIZE-NEXT:    return
// CHECK-CANONICALIZE-NEXT:  }
func.func @clean_region_branch_op_remove_last_2_results_last_2_arguments_and_last_operand(%arg2: i1) -> (i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %live, %non_live, %non_live_0, %non_live_1 = scf.while (%arg3 = %c0, %arg4 = %c1, %arg10 = %c2) : (i32, i32, i32) -> (i32, i32, i32, i32) {
    %non_live_2 = arith.addi %arg10, %arg10 : i32
    %non_live_3 = func.call @identity(%arg10) : (i32) -> (i32)
    scf.condition(%arg2) %arg4, %arg3, %non_live_2, %non_live_3 : i32, i32, i32, i32
  } do {
  ^bb0(%arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32):
    %non_live_4 = arith.addi %arg7, %arg8 :i32
    scf.yield %arg5, %arg6, %non_live_4 : i32, i32, i32
  }
  return %live : i32
}
func.func private @identity(%arg1 : i32) -> (i32) {
  return %arg1 : i32
}

// -----

// The op isn't erased because it has memory effects but its unnecessary result
// is removed.
//
// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK-CANONICALIZE:       func.func @clean_region_branch_op_remove_result(%[[arg0:.*]]: index, %[[arg1:.*]]: memref<i32>) {
// CHECK-CANONICALIZE-NEXT:    scf.index_switch %[[arg0]]
// CHECK-CANONICALIZE-NEXT:    case 1 {
// CHECK-CANONICALIZE-NEXT:      %[[c10:.*]] = arith.constant 10
// CHECK-CANONICALIZE-NEXT:      memref.store %[[c10]], %[[arg1]][]
// CHECK-CANONICALIZE:           scf.yield
// CHECK-CANONICALIZE-NEXT:    }
// CHECK-CANONICALIZE-NEXT:    default {
// CHECK-CANONICALIZE:         }
// CHECK-CANONICALIZE-NEXT:    return
// CHECK-CANONICALIZE-NEXT:  }
func.func @clean_region_branch_op_remove_result(%arg0 : index, %arg1 : memref<i32>) {
  %non_live = scf.index_switch %arg0 -> i32
  case 1 {
    %c10 = arith.constant 10 : i32
    memref.store %c10, %arg1[] : memref<i32>
    scf.yield %c10 : i32
  }
  default {
    %c11 = arith.constant 11 : i32
    scf.yield %c11 : i32
  }
  return
}

// -----

// The simple ops which don't have memory effects or live results get removed.
// %arg5 doesn't get removed from the @main even though it isn't live because
// the signature of a public function is always left untouched.
//
// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK:       func.func private @clean_simple_ops(%[[arg0:.*]]: i32, %[[arg1:.*]]: memref<i32>)
// CHECK-NEXT:    %[[live_0:.*]] = arith.addi %[[arg0]], %[[arg0]]
// CHECK-NEXT:    %[[c2:.*]] = arith.constant 2
// CHECK-NEXT:    %[[live_1:.*]] = arith.muli %[[live_0]], %[[c2]]
// CHECK-NEXT:    %[[c3:.*]] = arith.constant 3
// CHECK-NEXT:    %[[live_2:.*]] = arith.addi %[[arg0]], %[[c3]]
// CHECK-NEXT:    memref.store %[[live_2]], %[[arg1]][]
// CHECK-NEXT:    return %[[live_1]]
// CHECK-NEXT:  }
// CHECK:       func.func @main(%[[arg3:.*]]: i32, %[[arg4:.*]]: memref<i32>, %[[arg5:.*]]
// CHECK-NEXT:    %[[live:.*]] = call @clean_simple_ops(%[[arg3]], %[[arg4]])
// CHECK-NEXT:    return %[[live]]
// CHECK-NEXT:  }
func.func private @clean_simple_ops(%arg0 : i32, %arg1 : memref<i32>, %arg2 : i32) -> (i32, i32, i32, i32) {
  %live_0 = arith.addi %arg0, %arg0 : i32
  %c2 = arith.constant 2 : i32
  %live_1 = arith.muli %live_0, %c2 : i32
  %non_live_1 = arith.addi %live_1, %live_0 : i32
  %non_live_2 = arith.constant 7 : i32
  %non_live_3 = arith.subi %arg0, %non_live_1 : i32
  %c3 = arith.constant 3 : i32
  %live_2 = arith.addi %arg0, %c3 : i32
  memref.store %live_2, %arg1[] : memref<i32>
  return %live_1, %non_live_1, %non_live_2, %non_live_3 : i32, i32, i32, i32
}

func.func @main(%arg3 : i32, %arg4 : memref<i32>, %arg5 : i32) -> (i32) {
  %live, %non_live_1, %non_live_2, %non_live_3 = func.call @clean_simple_ops(%arg3, %arg4, %arg5) : (i32, memref<i32>, i32) -> (i32, i32, i32, i32)
  return %live : i32
}

// -----

// The scf.while op has no memory effects and its result isn't live.
//
// Note that this cleanup cannot be done by the `canonicalize` pass.
//
// CHECK-LABEL: func.func private @clean_region_branch_op_erase_it() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:       func.func @main(%[[arg3:.*]]: i32, %[[arg4:.*]]: i1) {
// CHECK-NEXT:    call @clean_region_branch_op_erase_it() : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func private @clean_region_branch_op_erase_it(%arg0 : i32, %arg1 : i1) -> (i32) {
  %non_live = scf.while (%arg2 = %arg0) : (i32) -> (i32) {
    scf.condition(%arg1) %arg2 : i32
  } do {
  ^bb0(%arg2: i32):
    scf.yield %arg2 : i32
  }
  return %non_live : i32
}

func.func @main(%arg3 : i32, %arg4 : i1) {
  %non_live_0 = func.call @clean_region_branch_op_erase_it(%arg3, %arg4) : (i32, i1) -> (i32)
  return
}

// -----

// The scf.if operation represents an if-then-else construct for conditionally
// executing two regions of code. The 'the' region has exactly 1 block, and
// the 'else' region may have 0 or 1 block. This case is to ensure 'else' region
// with 0 block not crash.

// CHECK-LABEL: func.func @clean_region_branch_op_with_empty_region
func.func @clean_region_branch_op_with_empty_region(%arg0: i1, %arg1: memref<f32>) {
  %cst = arith.constant 1.000000e+00 : f32
  scf.if %arg0 {
    memref.store %cst, %arg1[] : memref<f32>
  }
  return
}

// -----

#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
func.func @kernel(%arg0: memref<18xf32>) {
  %c1 = arith.constant 1 : index
  %c18 = arith.constant 18 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c18, %arg10 = %c18, %arg11 = %c18) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    %c1_0 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %cst_2 = arith.constant 25.4669495 : f32
    %6 = affine.apply #map(%arg3)[%c1_0, %c0_1]
    memref.store %cst_2, %arg0[%6] : memref<18xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  return
}

// CHECK-LABEL: func.func @kernel(%arg0: memref<18xf32>) {
// CHECK: gpu.launch blocks
// CHECK: memref.store
// CHECK-NEXT: gpu.terminator

// -----


// CHECK-LABEL: llvm_unreachable
// CHECK-LABEL: @fn_with_llvm_unreachable
// CHECK-LABEL: @main
// CHECK: llvm.return
module @llvm_unreachable {
  func.func private @fn_with_llvm_unreachable(%arg0: tensor<4x4xf32>) -> tensor<4x4xi1> {
    llvm.unreachable
  }
  func.func private @main(%arg0: tensor<4x4xf32>) {
    %0 = call @fn_with_llvm_unreachable(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xi1>
    llvm.return
  }
}

// CHECK: func.func private @no_block_func_declaration()
func.func private @no_block_func_declaration() -> ()

// -----

// CHECK: llvm.func @no_block_external_func()
llvm.func @no_block_external_func() attributes {sym_visibility = "private"}

// -----

// Check that yielded values aren't incorrectly removed in gpu regions
gpu.module @test_module_3 {
  gpu.func @gpu_all_reduce_region() {
    %arg0 = arith.constant 1 : i32
    %result = gpu.all_reduce %arg0 uniform {
    ^bb(%lhs : i32, %rhs : i32):
      %xor = arith.xori %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    } : (i32) -> (i32)
    gpu.return
  }
}

// CHECK-LABEL: func @gpu_all_reduce_region()
// CHECK: %[[yield:.*]] = arith.xori %{{.*}}, %{{.*}} : i32
// CHECK: gpu.yield %[[yield]] : i32

// -----

// Check that yielded values aren't incorrectly removed in linalg regions
module {
  func.func @linalg_red_add(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (0)>],
      iterator_types = ["reduction"]
    } ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      %2 = arith.subf %1, %out : f32 // this should still be removed
      linalg.yield %1 : f32
    } -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// CHECK-LABEL: func @linalg_red_add
// CHECK: %[[yield:.*]] = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: linalg.yield %[[yield]] : f32
// CHECK-NOT: arith.subf


// -----

// check that ops with zero operands are correctly handled

module {
  func.func @test_zero_operands(%I: memref<10xindex>, %I2: memref<10xf32>) {
    %v0 = arith.constant 0 : index
    %result = memref.alloca_scope -> index {
      %c = arith.addi %v0, %v0 : index
      memref.store %c, %I[%v0] : memref<10xindex>
      memref.alloca_scope.return %c: index
    }
    func.return
  }
}

// CHECK-CANONICALIZE-LABEL: func @test_zero_operands
// CHECK-CANONICALIZE-NEXT:    %[[c0:.*]] = arith.constant 0
// CHECK-CANONICALIZE-NEXT:    memref.store %[[c0]]
// CHECK-CANONICALIZE-NOT:     memref.alloca_scope.return

// -----

// CHECK-LABEL: func.func @test_atomic_yield
func.func @test_atomic_yield(%I: memref<10xf32>, %idx : index) {
  // CHECK: memref.generic_atomic_rmw
  %x = memref.generic_atomic_rmw %I[%idx] : memref<10xf32> {
  ^bb0(%current_value : f32):
    // CHECK: arith.constant
    %c1 = arith.constant 1.0 : f32
    // CHECK: memref.atomic_yield
    memref.atomic_yield %c1 : f32
  }
  func.return
}

// -----

// CHECK-LABEL: module @return_void_with_unused_argument
module @return_void_with_unused_argument {
  // CHECK-LABEL: func.func private @fn_return_void_with_unused_argument
  // CHECK-SAME: (%[[ARG0_FN:.*]]: i32)
  func.func private @fn_return_void_with_unused_argument(%arg0: i32, %arg1: memref<4xi32>) -> () {
    %sum = arith.addi %arg0, %arg0 : i32
    %c0 = arith.constant 0 : index
    %buf = memref.alloc() : memref<1xi32>
    memref.store %sum, %buf[%c0] : memref<1xi32>
    return
  }
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%[[ARG0_MAIN:.*]]: i32)
  // CHECK: call @fn_return_void_with_unused_argument(%[[ARG0_MAIN]]) : (i32) -> ()
  func.func @main(%arg0: i32) -> memref<4xi32> {
    %unused = memref.alloc() : memref<4xi32>
    call @fn_return_void_with_unused_argument(%arg0, %unused) : (i32, memref<4xi32>) -> ()
    return %unused : memref<4xi32>
  }
}

// -----

// CHECK-LABEL: module @dynamically_unreachable
module @dynamically_unreachable {
  func.func @dynamically_unreachable() {
    // This value is used by an operation in a dynamically unreachable block.
    %zero = arith.constant 0 : i64

    // Dataflow analysis knows from the constant condition that
    // ^bb1 is unreachable
    %false = arith.constant false
    cf.cond_br %false, ^bb1, ^bb4
  ^bb1:
    // This unreachable operation should be removed.
    // CHECK-NOT: arith.cmpi
    %3 = arith.cmpi eq, %zero, %zero : i64
    cf.br ^bb1
  ^bb4:
    return
  }
}

// CHECK-LABEL: module @last_block_not_exit
module @last_block_not_exit {
  // return value can be removed because it's private.
  func.func private @terminated_with_condbr(%arg0: i1, %arg1: i1) -> i1 {
    %true = arith.constant true
    %false = arith.constant false
    cf.cond_br %arg0, ^bb1(%false : i1), ^bb2
  ^bb1(%1: i1):  // 2 preds: ^bb0, ^bb2
    return %1 : i1
  ^bb2:  // pred: ^bb3
    cf.cond_br %arg1, ^bb1(%false : i1), ^bb1(%true : i1)
  }

  func.func public @call_private_but_not_use() {
    %i0 = arith.constant 0: i1
    %i1 = arith.constant 1: i1
    call @terminated_with_condbr(%i0, %i1) : (i1, i1) -> i1
    func.return
  }
  // CHECK-LABEL: @call_private_but_not_use
  // CHECK: call @terminated_with_condbr(%false, %true) : (i1, i1)
}

// -----

// Test the elimination of function arguments.

// CHECK-LABEL: func private @single_parameter
// CHECK-SAME: () {
func.func private @single_parameter(%arg0: index) {
  return
}

// CHECK-LABEL: func.func private @mutl_parameter(
// CHECK-SAME: %[[ARG0:.*]]: index)
// CHECK: return %[[ARG0]]
func.func private @mutl_parameter(%arg0: index, %arg1: index, %arg2: index) -> index {
  return %arg1 : index
}

// CHECK-LABEL: func private @eliminate_parameter
// CHECK-SAME: () {
func.func private @eliminate_parameter(%arg0: index, %arg1: index) {
  call @single_parameter(%arg0) : (index) -> ()
  return
}

// CHECK-LABEL: func @callee
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @callee(%arg0: index, %arg1: index, %arg2: index) -> index {
// CHECK: call @eliminate_parameter() : () -> ()
  call @eliminate_parameter(%arg0, %arg1) : (index, index) -> ()
// CHECK: call @mutl_parameter(%[[ARG1]]) : (index) -> index
  %res = call @mutl_parameter(%arg0, %arg1, %arg2) : (index, index, index) -> (index)
  return %res : index
}

// -----

// This test verifies that the induction variables in loops are not deleted, the loop has results.

// CHECK-LABEL: func @dead_value_loop_ivs
func.func @dead_value_loop_ivs_has_result(%lb: index, %ub: index, %step: index, %b: i1) -> i1 {
  %loop_ret = scf.for %iv = %lb to %ub step %step iter_args(%iter = %b) -> (i1) {
    cf.assert %b, "loop not dead"
    scf.yield %b : i1
  }
  return %loop_ret : i1
}

// -----

// This test verifies that the induction variables in loops are not deleted, the loop has no results.

// CHECK-LABEL: func @dead_value_loop_ivs_no_result
func.func @dead_value_loop_ivs_no_result(%lb: index, %ub: index, %step: index, %input: memref<?xf32>, %value: f32, %pos: index) {
  scf.for %iv = %lb to %ub step %step {
    memref.store %value, %input[%pos] : memref<?xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @op_block_have_dead_arg
func.func @op_block_have_dead_arg(%arg0: index, %arg1: index, %arg2: i1) {
  scf.execute_region {
    cf.cond_br %arg2, ^bb1(%arg0 : index), ^bb1(%arg1 : index)
  ^bb1(%0: index):
      scf.yield
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func private @remove_dead_branch_op()
// CHECK-NEXT:    ub.unreachable
// CHECK-NEXT:  ^{{.*}}:
// CHECK-NEXT:    return
// CHECK-NEXT:  ^{{.*}}:
// CHECK-NEXT:    return
func.func private @remove_dead_branch_op(%c: i1, %arg0: i64, %arg1: i64) -> (i64) {
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  return %arg0 : i64
^bb2:
  return %arg1 : i64
}

// -----

// CHECK-LABEL: func @affine_loop_no_use_iv_has_side_effect_op
func.func @affine_loop_no_use_iv_has_side_effect_op() {
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<10xindex>
  affine.for %arg0 = 0 to 79 {
    memref.store %c1, %alloc[%c1] : memref<10xindex>
  }
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<10xindex>
// CHECK: affine.for %[[VAL_0:.*]] = 0 to 79 {
// CHECK:   memref.store %[[C1]], %[[ALLOC]]{{\[}}%[[C1]]] : memref<10xindex>
// CHECK: }
  return
}

// -----

// CHECK-LABEL: func @scf_while_dead_iter_args()
// CHECK:         %[[c5:.*]] = arith.constant 5 : i32
// CHECK:         %[[while:.*]]:2 = scf.while (%[[arg0:.*]] = %[[c5]]) : (i32) -> (i32, i32) {
// CHECK:           vector.print %[[arg0]]
// CHECK:           %[[cmpi:.*]] = arith.cmpi
// CHECK:           %[[p0:.*]] = ub.poison : i32
// CHECK:           scf.condition(%[[cmpi]]) %[[arg0]], %[[p0]]
// CHECK:         } do {
// CHECK:         ^bb0(%[[arg1:.*]]: i32, %[[arg2:.*]]: i32):
// CHECK:           %[[p1:.*]] = ub.poison : i32
// CHECK:           scf.yield %[[p1]]
// CHECK:         }
// CHECK:         return %[[while]]#0

// CHECK-CANONICALIZE-LABEL: func @scf_while_dead_iter_args()
// CHECK-CANONICALIZE:         %[[c5:.*]] = arith.constant 5 : i32
// CHECK-CANONICALIZE:         %[[while:.*]] = scf.while (%[[arg0:.*]] = %[[c5]]) : (i32) -> i32 {
// CHECK-CANONICALIZE:           vector.print %[[arg0]]
// CHECK-CANONICALIZE:           %[[cmpi:.*]] = arith.cmpi
// CHECK-CANONICALIZE:           scf.condition(%[[cmpi]]) %[[arg0]]
// CHECK-CANONICALIZE:         } do {
// CHECK-CANONICALIZE:         ^bb0(%[[arg1:.*]]: i32):
// CHECK-CANONICALIZE:           %[[p0:.*]] = ub.poison : i32
// CHECK-CANONICALIZE:           scf.yield %[[p0]]
// CHECK-CANONICALIZE:         }
// CHECK-CANONICALIZE:         return %[[while]]
func.func @scf_while_dead_iter_args() -> i32 {
  %c5 = arith.constant 5 : i32
  %result:2 = scf.while (%arg0 = %c5) : (i32) -> (i32, i32) {
    vector.print %arg0 : i32
    // Note: This condition is always "false". (And the liveness analysis
    // can figure that out.)
    %cmp2 = arith.cmpi slt, %arg0, %c5 : i32
    scf.condition(%cmp2) %arg0, %arg0 : i32, i32
  } do {
  ^bb0(%arg1: i32, %arg2: i32):
    %x = scf.execute_region -> i32 {
      scf.yield %arg2 : i32
    }
    scf.yield %x : i32
  }
  return %result#0 : i32
}
