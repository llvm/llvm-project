// RUN: mlir-opt %s -acc-specialize-for-host | FileCheck %s

// Recipe definitions
acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

acc.firstprivate.recipe @firstprivatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
} copy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
}

acc.reduction.recipe @reduction_add_memref_i32 : memref<i32> reduction_operator <add> init {
^bb0(%arg0: memref<i32>):
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.alloca() : memref<i32>
  memref.store %c0_i32, %0[] : memref<i32>
  acc.yield %0 : memref<i32>
} combiner {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  %1 = memref.load %arg1[] : memref<i32>
  %2 = arith.addi %0, %1 : i32
  memref.store %2, %arg0[] : memref<i32>
  acc.yield %arg0 : memref<i32>
}

//===----------------------------------------------------------------------===//
// Orphan data entry ops - replaced with var
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_private func(@private) seq
// CHECK-LABEL: func.func @private
// CHECK-NOT:   acc.private
func.func @private(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_private]>} {
  %c0 = arith.constant 0 : i32
  %0 = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

acc.routine @acc_routine_cache func(@cache) seq
// CHECK-LABEL: func.func @cache
// CHECK-NOT:   acc.cache
func.func @cache(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_cache]>} {
  %c0 = arith.constant 0 : i32
  %0 = acc.cache varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

//===----------------------------------------------------------------------===//
// Orphan atomic operations - converted to load/store
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_atomic func(@orphan_atomic_update) seq
// CHECK-LABEL: func.func @orphan_atomic_update
// CHECK-NOT:   acc.atomic.update
// CHECK:       memref.load
// CHECK:       arith.addi
// CHECK:       memref.store
func.func @orphan_atomic_update(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_atomic]>} {
  acc.atomic.update %arg0 : memref<i32> {
  ^bb0(%arg1: i32):
    %c1 = arith.constant 1 : i32
    %1 = arith.addi %arg1, %c1 : i32
    acc.yield %1 : i32
  }
  return
}

acc.routine @acc_routine_atomic_read func(@orphan_atomic_read) seq
// CHECK-LABEL: func.func @orphan_atomic_read
// CHECK-NOT:   acc.atomic.read
// CHECK:       memref.copy %arg0, %arg1
func.func @orphan_atomic_read(%arg0 : memref<i32>, %arg1 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_atomic_read]>} {
  acc.atomic.read %arg1 = %arg0 : memref<i32>, memref<i32>, i32
  return
}

acc.routine @acc_routine_atomic_write func(@orphan_atomic_write) seq
// CHECK-LABEL: func.func @orphan_atomic_write
// CHECK-NOT:   acc.atomic.write
// CHECK:       memref.store %arg1, %arg0[]
func.func @orphan_atomic_write(%arg0 : memref<i32>, %arg1 : i32) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_atomic_write]>} {
  acc.atomic.write %arg0 = %arg1 : memref<i32>, i32
  return
}

acc.routine @acc_routine_atomic_capture func(@orphan_atomic_capture) seq
// CHECK-LABEL: func.func @orphan_atomic_capture
// CHECK-NOT:   acc.atomic.capture
// CHECK:       memref.copy %arg0, %arg1
// CHECK:       [[LOAD:%.*]] = memref.load %arg0[]
// CHECK:       [[INC:%.*]] = arith.addi [[LOAD]]
// CHECK:       memref.store [[INC]], %arg0[]
func.func @orphan_atomic_capture(%arg0 : memref<i32>, %arg1 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_atomic_capture]>} {
  %c1_i32 = arith.constant 1 : i32
  acc.atomic.capture {
    acc.atomic.read %arg1 = %arg0 : memref<i32>, memref<i32>, i32
    acc.atomic.update %arg0 : memref<i32> {
    ^bb0(%v: i32):
      %r = arith.addi %v, %c1_i32 : i32
      acc.yield %r : i32
    }
    acc.terminator
  }
  return
}

//===----------------------------------------------------------------------===//
// Negative tests - ops that should NOT be converted
//===----------------------------------------------------------------------===//

// acc.private attached to acc.parallel should NOT be removed
acc.routine @acc_routine_private_parallel func(@private_attached_to_parallel) seq
// CHECK-LABEL: func.func @private_attached_to_parallel
// CHECK:       acc.private
// CHECK:       acc.parallel
func.func @private_attached_to_parallel(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_private_parallel]>} {
  %0 = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  acc.parallel private(%0 : memref<i32>) {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %0[] : memref<i32>
    acc.yield
  }
  return
}

// acc.atomic.update inside acc.parallel should NOT be converted
acc.routine @acc_routine_atomic_parallel func(@atomic_inside_parallel) seq
// CHECK-LABEL: func.func @atomic_inside_parallel
// CHECK:       acc.parallel
// CHECK:       acc.atomic.update
func.func @atomic_inside_parallel(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_atomic_parallel]>} {
  acc.parallel {
    acc.atomic.update %arg0 : memref<i32> {
    ^bb0(%arg1: i32):
      %c1 = arith.constant 1 : i32
      %1 = arith.addi %arg1, %c1 : i32
      acc.yield %1 : i32
    }
    acc.yield
  }
  return
}

// acc.loop inside acc.parallel should NOT be converted
acc.routine @acc_routine_loop_parallel func(@loop_inside_parallel) seq
// CHECK-LABEL: func.func @loop_inside_parallel
// CHECK:       acc.parallel
// CHECK:       acc.loop
func.func @loop_inside_parallel(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_loop_parallel]>} {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  acc.parallel {
    acc.loop control(%iv : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
      %c5 = arith.constant 5 : i32
      memref.store %c5, %arg0[] : memref<i32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

//===----------------------------------------------------------------------===//
// Positive tests - orphan ops attached to orphan loop (both should convert)
//===----------------------------------------------------------------------===//

// acc.private attached to orphan acc.loop - BOTH should be removed
acc.routine @acc_routine_private_loop func(@private_attached_to_loop) seq
// CHECK-LABEL: func.func @private_attached_to_loop
// CHECK-NOT:   acc.private
// CHECK-NOT:   acc.loop
// CHECK:       scf.for
func.func @private_attached_to_loop(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_private_loop]>} {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %0 = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  acc.loop private(%0 : memref<i32>) control(%iv : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
    %c1_i32 = arith.constant 1 : i32
    memref.store %c1_i32, %0[] : memref<i32>
    acc.yield
  } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
  return
}

//===----------------------------------------------------------------------===//
// Orphan loop conversion tests
//===----------------------------------------------------------------------===//

// Orphan acc.loop should be converted to scf.for
acc.routine @acc_routine_loop func(@orphan_loop) seq
// CHECK-LABEL: func.func @orphan_loop
// CHECK-NOT:   acc.loop
// CHECK:       scf.for
func.func @orphan_loop(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_loop]>} {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  acc.loop control(%iv : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
    memref.store %iv, %arg0[] : memref<i32>
    acc.yield
  } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
  return
}

// Nested orphan acc.loop should be converted to nested scf.for
acc.routine @acc_routine_nested_loop func(@nested_orphan_loop) seq
// CHECK-LABEL: func.func @nested_orphan_loop
// CHECK-NOT:   acc.loop
// CHECK:       scf.for
// CHECK:       scf.for
func.func @nested_orphan_loop(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_nested_loop]>} {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  acc.loop control(%iv0 : i32, %iv1 : i32) = (%c0, %c0 : i32, i32) to (%c10, %c10 : i32, i32) step (%c1, %c1 : i32, i32) {
    %sum = arith.addi %iv0, %iv1 : i32
    memref.store %sum, %arg0[] : memref<i32>
    acc.yield
  } attributes {inclusiveUpperbound = array<i1: true, true>, seq = [#acc.device_type<none>]}
  return
}

//===----------------------------------------------------------------------===//
// Unstructured orphan loop - converted to scf.execute_region
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_unstructured func(@orphan_unstructured_loop) seq
// CHECK-LABEL: func.func @orphan_unstructured_loop
// CHECK-NOT:   acc.loop
// CHECK-NOT:   acc.private
// CHECK:       scf.execute_region
// CHECK:       ^bb{{[0-9]+}}:
// CHECK:       cf.cond_br
// CHECK:       scf.yield
func.func @orphan_unstructured_loop(%arg0 : memref<32xi32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_unstructured]>} {
  %c32_i32 = arith.constant 32 : i32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %iter_var = memref.alloca() : memref<i32>
  %priv = acc.private varPtr(%iter_var : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  acc.loop private(%priv : memref<i32>) {
    %limit = memref.alloca() : memref<i32>
    memref.store %c32_i32, %limit[] : memref<i32>
    memref.store %c1_i32, %priv[] : memref<i32>
    cf.br ^bb1
  ^bb1:
    %count = memref.load %limit[] : memref<i32>
    %cond = arith.cmpi sgt, %count, %c0_i32 : i32
    cf.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    %idx = memref.load %priv[] : memref<i32>
    %idx_idx = arith.index_cast %idx : i32 to index
    %val = memref.load %arg0[%idx_idx] : memref<32xi32>
    %new_val = arith.divsi %val, %c2_i32 : i32
    memref.store %new_val, %arg0[%idx_idx] : memref<32xi32>
    %new_count = arith.subi %count, %c1_i32 : i32
    memref.store %new_count, %limit[] : memref<i32>
    %new_idx = arith.addi %idx, %c1_i32 : i32
    memref.store %new_idx, %priv[] : memref<i32>
    cf.br ^bb1
  ^bb3:
    acc.yield
  } attributes {independent = [#acc.device_type<none>], unstructured}
  return
}

//===----------------------------------------------------------------------===//
// Orphan loop with reduction - both converted
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_loop_reduction func(@orphan_loop_with_reduction) seq
// CHECK-LABEL: func.func @orphan_loop_with_reduction
// CHECK-NOT:   acc.loop
// CHECK-NOT:   acc.reduction
// CHECK-NOT:   acc.private
// CHECK:       scf.for
func.func @orphan_loop_with_reduction(%arg0 : memref<i32>, %arg1 : memref<100xi32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_loop_reduction]>} {
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  %iter_var = memref.alloca() : memref<i32>
  %red = acc.reduction varPtr(%arg0 : memref<i32>) recipe(@reduction_add_memref_i32) -> memref<i32>
  %priv = acc.private varPtr(%iter_var : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  acc.loop vector private(%priv : memref<i32>) reduction(%red : memref<i32>) control(%arg2 : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
    memref.store %arg2, %priv[] : memref<i32>
    %idx = memref.load %priv[] : memref<i32>
    %idx_cast = arith.index_cast %idx : i32 to index
    %elem = memref.load %arg1[%idx_cast] : memref<100xi32>
    %r_val = memref.load %arg0[] : memref<i32>
    %new_r = arith.addi %r_val, %elem : i32
    memref.store %new_r, %arg0[] : memref<i32>
    acc.yield
  } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
  return
}

//===----------------------------------------------------------------------===//
// Orphan loop with variable bounds
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_var_bounds func(@orphan_loop_variable_bounds) seq
// CHECK-LABEL: func.func @orphan_loop_variable_bounds
// CHECK-NOT:   acc.loop
// CHECK:       [[LB:%.*]] = memref.load %arg0[]
// CHECK:       [[UB:%.*]] = memref.load %arg1[]
// CHECK:       scf.for
func.func @orphan_loop_variable_bounds(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_var_bounds]>} {
  %c1 = arith.constant 1 : i32
  %lb = memref.load %arg0[] : memref<i32>
  %ub = memref.load %arg1[] : memref<i32>
  acc.loop vector control(%iv : i32) = (%lb : i32) to (%ub : i32) step (%c1 : i32) {
    memref.store %iv, %arg2[] : memref<i32>
    acc.yield
  } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
  return
}

//===----------------------------------------------------------------------===//
// Orphan loop between compute regions - only orphan converted
//===----------------------------------------------------------------------===//

acc.reduction.recipe @reduction_mul_memref_i32 : memref<i32> reduction_operator <mul> init {
^bb0(%arg0: memref<i32>):
  %c1_i32 = arith.constant 1 : i32
  %0 = memref.alloca() : memref<i32>
  memref.store %c1_i32, %0[] : memref<i32>
  acc.yield %0 : memref<i32>
} combiner {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  %1 = memref.load %arg1[] : memref<i32>
  %2 = arith.muli %0, %1 : i32
  memref.store %2, %arg0[] : memref<i32>
  acc.yield %arg0 : memref<i32>
}

// Orphan loop sandwiched between compute regions - only orphan should convert
// CHECK-LABEL: func.func @orphan_between_compute_regions
// CHECK:       acc.parallel
// CHECK:       acc.yield
// CHECK-NOT:   acc.private varPtr
// CHECK-NOT:   acc.reduction varPtr
// CHECK:       scf.for
// CHECK:       acc.parallel
func.func @orphan_between_compute_regions(%arg0 : memref<i32>, %arg1 : memref<8xi32>, %arg2 : memref<i32>) {
  %c2_i32 = arith.constant 2 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %iter_var = memref.alloca() : memref<i32>

  // First compute region - should NOT be converted
  acc.parallel combined(loop) {
    %priv1 = acc.private varPtr(%iter_var : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
    acc.loop combined(parallel) private(%priv1 : memref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%c8_i32 : i32) step (%c1_i32 : i32) {
      memref.store %iv, %priv1[] : memref<i32>
      %idx = arith.index_cast %iv : i32 to index
      memref.store %c1_i32, %arg1[%idx] : memref<8xi32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.yield
  }

  // Orphan loop - SHOULD be converted
  %priv_orphan = acc.private varPtr(%arg2 : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  %red_orphan = acc.reduction varPtr(%arg0 : memref<i32>) recipe(@reduction_mul_memref_i32) -> memref<i32>
  %priv_iv = acc.private varPtr(%iter_var : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  acc.loop private(%priv_orphan, %priv_iv : memref<i32>, memref<i32>) reduction(%red_orphan : memref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%c8_i32 : i32) step (%c1_i32 : i32) {
    memref.store %iv, %priv_iv[] : memref<i32>
    %idx = arith.index_cast %iv : i32 to index
    %elem = memref.load %arg1[%idx] : memref<8xi32>
    memref.store %elem, %priv_orphan[] : memref<i32>
    %t = memref.load %priv_orphan[] : memref<i32>
    %mul = arith.muli %t, %c2_i32 : i32
    memref.store %mul, %arg0[] : memref<i32>
    acc.yield
  } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

  // Second compute region - should NOT be converted
  acc.parallel combined(loop) {
    %priv2 = acc.private varPtr(%iter_var : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
    acc.loop combined(parallel) private(%priv2 : memref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%c8_i32 : i32) step (%c1_i32 : i32) {
      memref.store %iv, %priv2[] : memref<i32>
      %idx = arith.index_cast %iv : i32 to index
      memref.store %iv, %arg1[%idx] : memref<8xi32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}
