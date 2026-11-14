// RUN: mlir-opt %s -split-input-file -test-greedy-patterns="top-down max-iterations=1" -verify-diagnostics | FileCheck %s

// Start each rewrite in a nonentry block to populate the reachability cache
// before changing the CFG, even when entry-block queries bypass the cache.

// A rewrite can create an unreachable block after the iteration's initial
// unreachable-block sweep. Its observer must not be processed.
// CHECK-LABEL: func.func @insert_unreachable
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @insert_unreachable() {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite append_unreachable
  cf.br ^exit
^exit:
  return
}

// -----

// A block connected before the rewrite returns is processed in this iteration,
// without relying on a fresh reachability cache in the next iteration.
// CHECK-LABEL: func.func @connect_before_return
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @connect_before_return(%cond: i1) {
  cf.br ^start
^start:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite append_reachable %cond
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
  test.greedy_cfg_rewrite insert_entry
  test.greedy_cfg_rewrite observe
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
  test.greedy_cfg_rewrite move_entry
  test.greedy_cfg_rewrite observe
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
  test.greedy_cfg_rewrite move_entry_away
  test.greedy_cfg_rewrite observe
  return
}

// -----

// The first nested op populates the ancestor cache, then moves its two-region
// owner into an unreachable block. The observer must see the invalidation.
// CHECK-LABEL: func.func @move_nested_unreachable
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @move_nested_unreachable() {
  cf.br ^start
^start:
  scf.execute_region {
    scf.execute_region {
      test.greedy_cfg_rewrite move_parent
      test.greedy_cfg_rewrite observe
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
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @disconnect_nested() {
  cf.br ^body
^body:
  scf.execute_region {
    test.greedy_cfg_rewrite disconnect_parent
    test.greedy_cfg_rewrite observe
    scf.yield
  }
  cf.br ^exit
^exit:
  return
}

// -----

// An in-place successor change can disconnect a block.
// CHECK-LABEL: func.func @redirect
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @redirect() {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite redirect
  cf.br ^body
^body:
  test.greedy_cfg_rewrite observe
  cf.br ^exit
^exit:
  return
}

// -----

// Populate the source cache, then move and erase a nonentry block in one
// rewrite. The next query must not inspect the erased block in that cache.
// CHECK-LABEL: func.func @move_then_erase
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @move_then_erase() {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite move_erase
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
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
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @merge() {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite merge
  cf.br ^body
^body:
  cf.br ^exit
^exit:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// Merging a block and dropping one of its successors leaves ^dead without
// predecessors. Its observer must be skipped and ^exit must still be processed.
// CHECK-LABEL: func.func @merge_disconnect
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @merge_disconnect(%cond: i1) {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite merge_disconnect
  cf.br ^body
^body:
  cf.cond_br %cond, ^exit, ^dead
^dead:
  test.greedy_cfg_rewrite observe
  return
^exit:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// Erasing the entry block makes its successor the new entry. The cache is
// populated before the rewrite, and the observer must run in this iteration.
// CHECK-LABEL: func.func @erase_entry
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @erase_entry() {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite erase_entry
  cf.br ^exit
^exit:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// Two blocks lose their edge to ^x in one rewrite. ^x has no predecessors left
// and ^s, its successor, is still a successor of ^p, but the entry lost ^x as
// well. Deciding ^s from the edge of ^p alone would keep it cached as reachable
// and process its observer.
// CHECK-LABEL: func.func @double_redirect
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @double_redirect() {
  cf.br ^x
^x:
  cf.br ^s
^p:
  test.greedy_cfg_rewrite double_redirect
  cf.br ^x
^s:
  test.greedy_cfg_rewrite observe
  cf.br ^p
}

// -----

// ^c and ^l change their terminators in one rewrite: ^c branches past ^l to
// ^s, and ^l, left without predecessors, branches to a new block holding an
// observer. Applying the change of ^l before that of ^c would record the new
// block as reachable; the observer in ^s must be processed, the new one not.
// CHECK-LABEL: func.func @redirect_retarget
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @redirect_retarget() {
  cf.br ^c
^c:
  test.greedy_cfg_rewrite redirect_retarget
  cf.br ^l
^l:
  cf.br ^s
^s:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// An in-place fold that retargets a branch is reported by the driver, not by
// the rewriter. The disconnected block must be skipped and the block that
// remains reachable must be processed in this iteration.
// CHECK-LABEL: func.func @fold_disconnects
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @fold_disconnects() {
  cf.br ^start
^start:
  "test.br"()[^mid] {fold_to_last} : () -> ()
^mid:
  test.greedy_cfg_rewrite observe
  cf.br ^exit
^exit:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// The lost edge leads into a block that keeps a predecessor: itself. The
// blocks behind it cannot be decided incrementally, and the query for the
// observer must traverse the region.
// CHECK-LABEL: func.func @lose_edge_into_loop
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: return
func.func @lose_edge_into_loop(%cond: i1) {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite redirect
  cf.br ^loop
^loop:
  test.greedy_cfg_rewrite observe
  cf.cond_br %cond, ^loop, ^exit
^exit:
  return
}

// -----

// The first rewrite disconnects ^body, the second one adds an edge back to
// it. The observer in ^body is not re-enqueued by the reconnection, but it
// is still on the worklist and must be processed in this iteration.
// CHECK-LABEL: func.func @reconnect
// CHECK-NOT: test.greedy_cfg_rewrite
// CHECK: cf.cond_br %{{.*}}, ^[[EXIT:.*]], ^[[BODY:.*]]
// CHECK: ^[[BODY]]:
// CHECK-NEXT: cf.br ^[[EXIT]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: return
func.func @reconnect(%cond: i1) {
  cf.br ^start
^start:
  test.greedy_cfg_rewrite redirect
  test.greedy_cfg_rewrite reconnect %cond
  cf.br ^body
^body:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  cf.br ^exit
^exit:
  return
}
