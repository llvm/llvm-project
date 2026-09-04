// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -test-greedy-patterns="top-down max-iterations=1" -verify-diagnostics | FileCheck %s

// Start each rewrite in a nonentry block to populate the reachability cache
// before changing the CFG, even when entry-block queries bypass the cache.

// A rewrite can create an unreachable block after the iteration's initial
// unreachable-block sweep. Its observer must not be processed.
// CHECK-LABEL: func.func @insert_unreachable
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @insert_unreachable() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "unreachable"} : () -> ()
  cf.br ^exit
^exit:
  return
}

// -----

// A block connected before the rewrite returns is processed in this iteration,
// without relying on a fresh reachability cache in the next iteration.
// CHECK-LABEL: func.func @connect_before_return
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @connect_before_return(%cond: i1) {
  cf.br ^start
^start:
  // expected-remark @+1 {{processed reachable block}}
  "test.greedy_create_block"(%cond) {mode = "reachable"} : (i1) -> ()
  cf.br ^exit
^exit:
  return
}

// -----

// Inserting a new entry can disconnect the old entry without editing its
// terminator.
// CHECK-LABEL: func.func @insert_entry
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @insert_entry() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "entry"} : () -> ()
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  cf.br ^exit
^exit:
  return
}

// -----

// Moving an existing block to the front must also invalidate the old entry.
// CHECK-LABEL: func.func @move_entry
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @move_entry() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "move-entry"} : () -> ()
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  cf.br ^exit
^exit:
  return
}

// -----

// Moving the old entry away from the front also changes the traversal root.
// The rewrite inserts a return-only block after the entry, then moves the old
// entry to the end. No successor changes, but the observer becomes unreachable.
// CHECK-LABEL: func.func @move_entry_away
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @move_entry_away() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "move-entry-away"} : () -> ()
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  return
}

// -----

// The first nested op populates the ancestor cache, then moves its two-region
// owner into an unreachable block. The observer must see the invalidation.
// CHECK-LABEL: func.func @move_nested_unreachable
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @move_nested_unreachable() {
  cf.br ^start
^start:
  scf.execute_region {
    scf.execute_region {
      "test.greedy_create_block"() {mode = "move-parent"} : () -> ()
      "test.greedy_create_block"() {mode = "observe"} : () -> ()
      scf.yield
    }
    scf.yield
  }
  return
}

// -----

// A branch rewrite can disconnect an outer block while leaving its nested
// region in place. The observer must not use the cached ancestor result.
// CHECK-LABEL: func.func @disconnect_nested
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @disconnect_nested() {
  cf.br ^body
^body:
  scf.execute_region {
    "test.greedy_create_block"() {mode = "disconnect-parent"} : () -> ()
    "test.greedy_create_block"() {mode = "observe"} : () -> ()
    scf.yield
  }
  cf.br ^exit
^exit:
  return
}

// -----

// An in-place successor change can disconnect a block.
// CHECK-LABEL: func.func @redirect
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @redirect() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "redirect"} : () -> ()
  cf.br ^body
^body:
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  cf.br ^exit
^exit:
  return
}

// -----

// Populate the source cache, then move and erase a nonentry block in one
// rewrite. The next query must not inspect the erased block in that cache.
// CHECK-LABEL: func.func @move_then_erase
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @move_then_erase() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "move-erase"} : () -> ()
  // expected-remark @+1 {{processed reachable block}}
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  cf.br ^body
^body:
  cf.br ^exit
^exit:
  return
}

// -----

// Merging a block preserves reachability of its successors. The cache is
// populated before the rewrite, and the observer must run in this iteration.
// CHECK-LABEL: func.func @merge
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @merge() {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "merge"} : () -> ()
  cf.br ^body
^body:
  cf.br ^exit
^exit:
  // expected-remark @+1 {{processed reachable block}}
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  return
}

// -----

// Merging a block and dropping one of its successors must force a rescan:
// preserving reachability based only on the erased block would process ^dead.
// CHECK-LABEL: func.func @merge_disconnect
// CHECK-NOT: "test.greedy_create_block"
// CHECK: return
func.func @merge_disconnect(%cond: i1) {
  cf.br ^start
^start:
  "test.greedy_create_block"() {mode = "merge-disconnect"} : () -> ()
  cf.br ^body
^body:
  cf.cond_br %cond, ^exit, ^dead
^dead:
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  return
^exit:
  // expected-remark @+1 {{processed reachable block}}
  "test.greedy_create_block"() {mode = "observe"} : () -> ()
  return
}
