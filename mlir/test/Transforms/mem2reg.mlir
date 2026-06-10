// RUN: mlir-opt %s --pass-pipeline='builtin.module(any(mem2reg))' --split-input-file | FileCheck %s

// Verifies that allocators with multiple slots are handled properly.

// CHECK-LABEL: func.func @multi_slot_alloca
func.func @multi_slot_alloca() -> (i32, i32) {
  // CHECK-NOT: test.multi_slot_alloca
  %1, %2 = test.multi_slot_alloca : () -> (memref<i32>, memref<i32>)
  %3 = memref.load %1[] : memref<i32>
  %4 = memref.load %2[] : memref<i32>
  return %3, %4 : i32, i32
}

// -----

// Verifies that a multi slot allocator can be partially promoted.

func.func private @consumer(memref<i32>)

// CHECK-LABEL: func.func @multi_slot_alloca_only_second
func.func @multi_slot_alloca_only_second() -> (i32, i32) {
  // CHECK: %{{[[:alnum:]]+}} = test.multi_slot_alloca
  %1, %2 = test.multi_slot_alloca : () -> (memref<i32>, memref<i32>)
  func.call @consumer(%1) : (memref<i32>) -> ()
  %3 = memref.load %1[] : memref<i32>
  %4 = memref.load %2[] : memref<i32>
  return %3, %4 : i32, i32
}

// -----

// Checks that slots are not promoted if used in a graph region.

// CHECK-LABEL: test.isolated_graph_region
test.isolated_graph_region {
  // CHECK: %{{[[:alnum:]]+}} = test.multi_slot_alloca
  %slot = test.multi_slot_alloca : () -> (memref<i32>)
  memref.store %a, %slot[] : memref<i32>
  %a = memref.load %slot[] : memref<i32>
  "test.foo"() : () -> ()
}

// -----

// Verifies that block arguments of merge points are not abusively treated as
// the newly created block arguments. Here, ^merge has a pre-existing block
// argument (%genuine) and mem2reg adds a second one for the promoted slot. The
// slot arg then serves as the reaching definition for the follow-up merge point
// ^final. If the unused merge point propagation logic identified merge points
// by block rather than by specific block argument, it would confuse %genuine
// for the slot argument to be removed and thus not eliminate the slot which
// is unused. In other words, the genuine block argument, which is used, would
// mask that the actual slot argument is unused.

// CHECK-LABEL: func.func @merge_point_arg_not_confused
// CHECK-SAME: (%[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
// CHECK: cf.cond_br %[[COND]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK: ^[[BB1]]:
// CHECK:   cf.br ^[[MERGE:.*]](%[[A]] : i32)
// CHECK: ^[[BB2]]:
// CHECK:   cf.br ^[[MERGE]](%[[B]] : i32)
// CHECK: ^[[MERGE]](%[[GENUINE:.*]]: i32):
// CHECK:   cf.cond_br %[[COND]], ^[[BB3:.*]], ^[[BB4:.*]]
// CHECK: ^[[BB3]]:
// CHECK:   cf.br ^[[FINAL:.*]](%[[GENUINE]] : i32)
// CHECK: ^[[BB4]]:
// CHECK:   %[[DUMMY:.*]] = arith.constant 0 : i32
// CHECK:   cf.br ^[[FINAL]](%[[DUMMY]] : i32)
// CHECK: ^[[FINAL]](%[[FINAL_SLOT:.*]]: i32):
// CHECK:   return %[[FINAL_SLOT]] : i32
func.func @merge_point_arg_not_confused(%cond: i1, %a: i32, %b: i32) -> i32 {
  %alloca = memref.alloca() : memref<i32>
  memref.store %a, %alloca[] : memref<i32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  memref.store %b, %alloca[] : memref<i32>
  cf.br ^merge(%a : i32)
^bb2:
  cf.br ^merge(%b : i32)
^merge(%genuine: i32):
  cf.cond_br %cond, ^bb3, ^bb4
^bb3:
  memref.store %genuine, %alloca[] : memref<i32>
  cf.br ^final
^bb4:
  %dummy = arith.constant 0 : i32
  memref.store %dummy, %alloca[] : memref<i32>
  cf.br ^final
^final:
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check that a load inside an unknown region-bearing op prevents promotion.

// CHECK-LABEL: func.func @unknown_region_op_load
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %[[C5]], %[[ALLOCA]][]
// CHECK: "test.one_region_op"() ({
// CHECK:   %[[LOAD:.*]] = memref.load %[[ALLOCA]][]
// CHECK:   "test.finish"() : () -> ()
// CHECK: }) : () -> ()
func.func @unknown_region_op_load() {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  "test.one_region_op"() ({
    %load = memref.load %alloca[] : memref<i32>
    "test.finish"() : () -> ()
  }) : () -> ()
  return
}

// -----

// A cycle of merge points where both merge point block arguments are unused.
// merge1 branches to merge2, and merge2 branches back to merge1, so each
// merge point's reaching definition arg is used as a successor operand
// feeding the other. During removeUnusedItems, the successor operand erasure
// and block argument erasure must be performed in separate phases. Otherwise,
// regardless of iteration order, erasing either arg first will crash because
// the other's successor operand still uses it.

// CHECK-LABEL: func.func @cyclic_unused_merge_points
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK-NOT: memref.alloca
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: cf.br ^[[MERGE1:.*]]{{$}}
// CHECK: ^[[MERGE1]]:
// CHECK:   cf.cond_br %[[COND]], ^[[MERGE2:.*]], ^[[STORE:.*]]
// CHECK: ^[[STORE]]:
// CHECK:   cf.br ^[[MERGE2]]{{$}}
// CHECK: ^[[MERGE2]]:
// CHECK:   cf.cond_br %[[COND]], ^[[MERGE1]], ^[[EXIT:.*]]
// CHECK: ^[[EXIT]]:
// CHECK:   return %[[C0]] : i32
func.func @cyclic_unused_merge_points(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c0, %alloca[] : memref<i32>
  cf.br ^merge1
^merge1:
  cf.cond_br %cond, ^merge2, ^store
^store:
  memref.store %c1, %alloca[] : memref<i32>
  cf.br ^merge2
^merge2:
  cf.cond_br %cond, ^merge1, ^exit
^exit:
  return %c0 : i32
}

// -----

// This function contains a loop bb1 -> bb2 -> bb1, and to promote the alloca, we
// need to insert a poison value that gets used initially instead of the stored
// value in case %cond2 is false. This poison value becomes a bb arg in bb1 and
// bb2, so the entry block must also pass the poison value to bb1. Make sure the
// poison value is generated in the entry block where it dominates all uses.

// CHECK-LABEL: func.func @poison_insertion_point
// CHECK-NEXT: ub.poison
func.func @poison_insertion_point(%val: f64) {
  cf.br ^bb1
^bb1:
  %alloca = memref.alloca() : memref<f64>
  %cond1 = "test.get"() : () -> i1
  cf.cond_br %cond1, ^bb2, ^bb3
^bb2:
  %cond2 = "test.get"() : () -> i1
  scf.if %cond2 {
    memref.store %val, %alloca[] : memref<f64>
  }
  %reload = memref.load %alloca[] : memref<f64>
  "test.use"(%reload) : (f64) -> ()
  cf.br ^bb1
^bb3:
  return
}

// -----

// Verifies that mem2reg promotes a memory slot accessed through a transparent
// alias operation exposing itself via `getPromotableSlotAliases`. The
// conditional store on the alias in ^bb1 must be discovered as a defining
// block; otherwise, the merge point at ^bb2 would lack a block argument,
// silently dropping the conditional update.

// CHECK-LABEL: func.func @promotable_through_alias
// CHECK-SAME: (%[[A:.*]]: i32, %[[COND:.*]]: i1) -> i32
// CHECK-NOT: test.multi_slot_alloca
// CHECK-NOT: test.transparent_alias
// CHECK: %[[C42:.*]] = arith.constant 42 : i32
// CHECK: cf.cond_br %[[COND]], ^[[BB1:.*]], ^[[BB2:.*]](%[[C42]] : i32)
// CHECK: ^[[BB1]]:
// CHECK:   cf.br ^[[BB2]](%[[A]] : i32)
// CHECK: ^[[BB2]](%[[MERGE:.*]]: i32):
// CHECK:   return %[[MERGE]] : i32
func.func @promotable_through_alias(%a: i32, %cond: i1) -> i32 {
  %c42 = arith.constant 42 : i32
  %slot = test.multi_slot_alloca : () -> memref<i32>
  %alias = test.transparent_alias %slot : (memref<i32>) -> memref<i32>
  memref.store %c42, %alias[] : memref<i32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  memref.store %a, %alias[] : memref<i32>
  cf.br ^bb2
^bb2:
  %v = memref.load %alias[] : memref<i32>
  return %v : i32
}

// -----

// Type-changing transparent alias: the store and load access the slot as f32
// while the underlying allocation is i32. mem2reg materializes an
// `unrealized_conversion_cast` at the store (f32 → i32 via `projectAliasValueToSlotValue`)
// and at the load (i32 → f32 via `projectSlotValueToAliasValue`).

// CHECK-LABEL: func.func @promotable_through_cast_alias
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
// CHECK-NOT: test.multi_slot_alloca
// CHECK-NOT: test.transparent_cast_alias
// CHECK: %[[I32:.*]] = builtin.unrealized_conversion_cast %[[A]] : f32 to i32
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[I32]] : i32 to f32
// CHECK: return %{{.*}} : f32
func.func @promotable_through_cast_alias(%a: f32) -> f32 {
  %slot = test.multi_slot_alloca : () -> memref<i32>
  %alias = test.transparent_cast_alias %slot : (memref<i32>) -> memref<f32>
  memref.store %a, %alias[] : memref<f32>
  %v = memref.load %alias[] : memref<f32>
  return %v : f32
}

// -----

// Same as above with a conditional store across blocks. The merge-point
// block argument uses the root slot's element type (i32). Casts are inserted
// at the store sites (f32 → i32 via `projectAliasValueToSlotValue`) and the
// load site (i32 → f32 via `projectSlotValueToAliasValue`).

// CHECK-LABEL: func.func @promotable_through_cast_alias_blocks
// CHECK-SAME: (%[[A:.*]]: f32, %[[COND:.*]]: i1) -> f32
// CHECK-NOT: test.multi_slot_alloca
// CHECK-NOT: test.transparent_cast_alias
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_I32:.*]] = builtin.unrealized_conversion_cast %[[CST]] : f32 to i32
// CHECK: cf.cond_br %[[COND]], ^[[BB1:.*]], ^[[BB2:.*]](%[[CST_I32]] : i32)
// CHECK: ^[[BB1]]:
// CHECK:   %[[A_I32:.*]] = builtin.unrealized_conversion_cast %[[A]] : f32 to i32
// CHECK:   cf.br ^[[BB2]](%[[A_I32]] : i32)
// CHECK: ^[[BB2]](%[[MERGE:.*]]: i32):
// CHECK:   %[[MERGE_F32:.*]] = builtin.unrealized_conversion_cast %[[MERGE]] : i32 to f32
// CHECK:   return %[[MERGE_F32]] : f32
func.func @promotable_through_cast_alias_blocks(%a: f32, %cond: i1) -> f32 {
  %cst = arith.constant 1.0 : f32
  %slot = test.multi_slot_alloca : () -> memref<i32>
  %alias = test.transparent_cast_alias %slot : (memref<i32>) -> memref<f32>
  memref.store %cst, %alias[] : memref<f32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  memref.store %a, %alias[] : memref<f32>
  cf.br ^bb2
^bb2:
  %v = memref.load %alias[] : memref<f32>
  return %v : f32
}

// -----

// Regression test: the alias is defined in the parent region, but the store
// is in a nested region (`scf.if`). The new blocking use must be registered
// under the store's region; otherwise, `removeBlockingUses` fails the region
// invariant after `scf.if` rebuilds itself in `finalizePromotion`.

// CHECK-LABEL: func.func @promotable_through_alias_across_regions
// CHECK-SAME: (%[[COND:.*]]: i1, %[[A:.*]]: i32)
// CHECK-NOT: test.multi_slot_alloca
// CHECK-NOT: test.transparent_alias
// CHECK-NOT: memref.store
// CHECK: scf.if %[[COND]]
func.func @promotable_through_alias_across_regions(%cond: i1, %a: i32) {
  %slot = test.multi_slot_alloca : () -> memref<i32>
  %alias = test.transparent_alias %slot : (memref<i32>) -> memref<i32>
  scf.if %cond {
    memref.store %a, %alias[] : memref<i32>
  }
  return
}

// -----

// Mirror case: the alias is created *inside* `scf.if`, used to store an
// `f32` value through a type-changing alias, while the parent `i32` slot
// is read outside. The alias-to-slot projection (`f32` -> `i32`) must run
// *inside* the region (where the alias is alive) and the resulting `i32`
// value must be threaded out of `scf.if` via its `setupPromotion`/
// `finalizePromotion` hooks to feed the parent load.

// CHECK-LABEL: func.func @alias_inside_region_parent_read_outside
// CHECK-SAME: (%[[COND:.*]]: i1, %[[A:.*]]: f32, %[[INIT:.*]]: i32) -> i32
// CHECK-NOT: test.multi_slot_alloca
// CHECK-NOT: test.transparent_cast_alias
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i32)
// CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[A]] : f32 to i32
// CHECK:   scf.yield %[[CAST]] : i32
// CHECK: } else {
// CHECK:   scf.yield %[[INIT]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @alias_inside_region_parent_read_outside(%cond: i1, %a: f32,
                                                   %init: i32) -> i32 {
  %slot = test.multi_slot_alloca : () -> memref<i32>
  memref.store %init, %slot[] : memref<i32>
  scf.if %cond {
    %alias = test.transparent_cast_alias %slot : (memref<i32>) -> memref<f32>
    memref.store %a, %alias[] : memref<f32>
  }
  %v = memref.load %slot[] : memref<i32>
  return %v : i32
}

// -----

// Chained aliasers: an identity alias is aliased by a type-changing alias.
// The alias-map walk must follow both hops and project through each step.

// CHECK-LABEL: func.func @promotable_through_chained_aliases
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
// CHECK-NOT: test.multi_slot_alloca
// CHECK-NOT: test.transparent_alias
// CHECK-NOT: test.transparent_cast_alias
// CHECK: %[[I32:.*]] = builtin.unrealized_conversion_cast %[[A]] : f32 to i32
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[I32]] : i32 to f32
// CHECK: return %{{.*}} : f32
func.func @promotable_through_chained_aliases(%a: f32) -> f32 {
  %slot = test.multi_slot_alloca : () -> memref<i32>
  %alias1 = test.transparent_alias %slot : (memref<i32>) -> memref<i32>
  %alias2 = test.transparent_cast_alias %alias1 : (memref<i32>) -> memref<f32>
  memref.store %a, %alias2[] : memref<f32>
  %v = memref.load %alias2[] : memref<f32>
  return %v : f32
}

// -----

// Dual-alias case: a single aliaser op exposes two simultaneously usable
// aliases of the same parent slot (signless i32) at different signednesses
// (signed and unsigned i32). `getPromotableSlotAliases` populates two
// entries for that operand, both of which end up in the alias map. The
// store reaches the slot through the signed alias and the load reaches it
// through the unsigned alias.

// CHECK-LABEL: func.func @promotable_through_dual_alias
// CHECK-SAME: (%[[A:.*]]: si32) -> ui32
// CHECK-NOT: memref.alloca
// CHECK-NOT: test.transparent_dual_alias
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
// CHECK: %[[A_I32:.*]] = builtin.unrealized_conversion_cast %[[A]] : si32 to i32
// CHECK: %[[A_UI32:.*]] = builtin.unrealized_conversion_cast %[[A_I32]] : i32 to ui32
// CHECK: return %[[A_UI32]] : ui32
func.func @promotable_through_dual_alias(%a: si32) -> ui32 {
  %slot = memref.alloca() : memref<i32>
  %signed, %unsigned = test.transparent_dual_alias %slot
    : (memref<i32>) -> (memref<si32>, memref<ui32>)
  memref.store %a, %signed[] : memref<si32>
  %v = memref.load %unsigned[] : memref<ui32>
  return %v : ui32
}

// -----

// Partial aliasing: the parent slot stores a `complex<f32>` (a 2-tuple of
// `f32`), and the alias exposes one component as a `memref<f32>`.
// The alias-to-slot projection reconstructs the parent value by consuming the
// current reaching definition (modelled as a 2-input `unrealized_conversion_cast`:
// new sub-value + parent reaching def). The slot-to-alias projection extracts
// a component (1-input cast).

// CHECK-LABEL: func.func @promotable_through_partial_alias
// CHECK-SAME: (%[[X:.*]]: f32) -> f32
// CHECK-NOT: memref.alloca
// CHECK-NOT: test.partial_alias
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
// CHECK: %[[POISON:.*]] = ub.poison : complex<f32>
// CHECK: %[[NEW:.*]] = builtin.unrealized_conversion_cast %[[X]], %[[POISON]] : f32, complex<f32> to complex<f32>
// CHECK: %[[R:.*]] = builtin.unrealized_conversion_cast %[[NEW]] : complex<f32> to f32
// CHECK: return %[[R]] : f32
func.func @promotable_through_partial_alias(%x: f32) -> f32 {
  %slot = memref.alloca() : memref<complex<f32>>
  %alias = test.partial_alias %slot : (memref<complex<f32>>) -> memref<f32>
  memref.store %x, %alias[] : memref<f32>
  %v = memref.load %alias[] : memref<f32>
  return %v : f32
}
