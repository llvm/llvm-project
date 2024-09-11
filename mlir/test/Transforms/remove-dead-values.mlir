// RUN: mlir-opt %s -remove-dead-values -split-input-file -verify-diagnostics | FileCheck %s

// The IR remains untouched because of the presence of a non-function-like
// symbol op (module @dont_touch_unacceptable_ir).
//
// expected-error @+1 {{cannot optimize an IR with non-function symbol ops, non-call symbol user ops or branch ops}}
module @dont_touch_unacceptable_ir {
  func.func @has_cleanable_simple_op(%arg0 : i32) {
    %non_live = arith.addi %arg0, %arg0 : i32
    return
  }
}

// -----

// The IR remains untouched because of the presence of a branch op `cf.cond_br`.
//
func.func @dont_touch_unacceptable_ir_has_cleanable_simple_op_with_branch_op(%arg0: i1) {
  %non_live = arith.constant 0 : i32
  // expected-error @+1 {{cannot optimize an IR with non-function symbol ops, non-call symbol user ops or branch ops}}
  cf.cond_br %arg0, ^bb1(%non_live : i32), ^bb2(%non_live : i32)
^bb1(%non_live_0 : i32):
  cf.br ^bb3
^bb2(%non_live_1 : i32):
  cf.br ^bb3
^bb3:
  return
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
//
// CHECK:       func.func @clean_region_branch_op_dont_remove_first_2_results_but_remove_first_operand(%[[arg0:.*]]: i1, %[[arg1:.*]]: i32, %[[arg2:.*]]: i32) -> i32 {
// CHECK-NEXT:    %[[live_and_non_live:.*]]:2 = scf.while (%[[arg4:.*]] = %[[arg2]]) : (i32) -> (i32, i32) {
// CHECK-NEXT:      %[[live_0:.*]] = arith.addi %[[arg4]], %[[arg4]]
// CHECK-NEXT:      scf.condition(%arg0) %[[live_0]], %[[arg4]] : i32, i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[arg5:.*]]: i32, %[[arg6:.*]]: i32):
// CHECK-NEXT:      %[[live_1:.*]] = arith.addi %[[arg6]], %[[arg6]]
// CHECK-NEXT:      scf.yield %[[live_1]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[live_and_non_live]]#0
// CHECK-NEXT:  }
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
// CHECK:       func.func @clean_region_branch_op_remove_last_2_results_last_2_arguments_and_last_operand(%[[arg2:.*]]: i1) -> i32 {
// CHECK-NEXT:    %[[c0:.*]] = arith.constant 0
// CHECK-NEXT:    %[[c1:.*]] = arith.constant 1
// CHECK-NEXT:    %[[live_and_non_live:.*]]:2 = scf.while (%[[arg3:.*]] = %[[c0]], %[[arg4:.*]] = %[[c1]]) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:      func.call @identity() : () -> ()
// CHECK-NEXT:      scf.condition(%[[arg2]]) %[[arg4]], %[[arg3]] : i32, i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[arg5:.*]]: i32, %[[arg6:.*]]: i32):
// CHECK-NEXT:      scf.yield %[[arg5]], %[[arg6]] : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[live_and_non_live]]#0 : i32
// CHECK-NEXT:  }
// CHECK:       func.func private @identity() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
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
// CHECK:       func.func @clean_region_branch_op_remove_result(%[[arg0:.*]]: index, %[[arg1:.*]]: memref<i32>) {
// CHECK-NEXT:    scf.index_switch %[[arg0]]
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      %[[c10:.*]] = arith.constant 10
// CHECK-NEXT:      memref.store %[[c10]], %[[arg1]][]
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
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

// CHECK: func.func private @no_block_func_declaration()
func.func private @no_block_func_declaration() -> ()
