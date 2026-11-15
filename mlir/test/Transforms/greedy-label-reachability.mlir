// RUN: mlir-opt %s -split-input-file -test-greedy-patterns="top-down max-iterations=1" | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: mlir-opt %s -split-input-file -test-greedy-patterns="top-down max-iterations=1 rewrite-after-labels" -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,ON

// Control enters a block that starts with a label-like op without a CFG edge.
// That block and the blocks it reaches are never erased as unreachable, but
// their operations are processed only with label reachability. A block
// without predecessors or label is still erased.
// CHECK-LABEL: func.func @label_block
// CHECK:         test.label "L"
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    cf.br ^[[NEXT:bb[0-9]+]]
// CHECK-NEXT:  ^[[NEXT]]:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @label_block() {
  return
^label:
  test.label "L"
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  cf.br ^next
^next:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
^dead:
  test.greedy_cfg_rewrite observe
  return
}

// -----

// A label nested in a region op makes the block holding that op reachable
// from the op onward: control that enters the label continues after the op.
// The operations before the op stay unreachable.
// CHECK-LABEL: func.func @nested_label
// CHECK:         test.greedy_cfg_rewrite observe
// CHECK-NEXT:    "test.ssacfg_region"() ({
// CHECK-NEXT:      "test.return"() : () -> ()
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      test.label "M"
// OFF-NEXT:        test.greedy_cfg_rewrite observe
// CHECK-NEXT:      "test.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
func.func @nested_label() {
  return
^holder:
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "M"
    // expected-remark @+1 {{processed reachable block}}
    test.greedy_cfg_rewrite observe
    "test.return"() : () -> ()
  }) : () -> ()
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// A block holding a nested label that a reachable block branches to, here the
// block itself, is entered at its start, so all its operations are reachable.
// CHECK-LABEL: func.func @nested_label_loop
// CHECK:       ^bb1:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    "test.ssacfg_region"
// CHECK:         cf.cond_br %{{.*}}, ^bb1, ^[[EXIT:bb[0-9]+]]
// CHECK-NEXT:  ^[[EXIT]]:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
func.func @nested_label_loop(%c: i1) {
  return
^holder:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "M"
    "test.return"() : () -> ()
  }) : () -> ()
  cf.cond_br %c, ^holder, ^exit
^exit:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}

// -----

// A label inside an op that is isolated from above does not make the block
// holding that op reachable, so the block is erased.
// CHECK-LABEL: func.func @isolated_label
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @isolated_label() {
  return
^holder:
  test.isolated_regions {
    "test.return"() : () -> ()
  ^label:
    test.label "N"
    "test.return"() : () -> ()
  }
  return
}

// The same holds when the isolated op is nested in another region op.
// CHECK-LABEL: func.func @isolated_label_nested
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @isolated_label_nested() {
  return
^holder:
  "test.ssacfg_region"() ({
    test.isolated_regions {
      "test.return"() : () -> ()
    ^label:
      test.label "N"
      "test.return"() : () -> ()
    }
    "test.return"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Erasing a label makes its block and the blocks behind it unreachable, so
// their observers are skipped and region simplification erases them at the
// end of the iteration. The observer in ^warm2 queries the cache, which then
// holds the labeled blocks, before the erasure. Once it is processed, ^warm2
// and ^start are identical and merge.
// CHECK-LABEL: func.func @erase_label
// CHECK:         test.label "W"
// CHECK-NEXT:    cf.br ^bb2
// CHECK-NEXT:  ^bb2:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
// OFF-NEXT:    ^bb3:
// OFF-NEXT:      return
// CHECK-NEXT:  }
func.func @erase_label() {
  cf.br ^start
^warm:
  test.label "W"
  cf.br ^warm2
^warm2:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
^start:
  test.greedy_cfg_rewrite erase_next_label
  return
^label:
  test.label "L"
  test.greedy_cfg_rewrite observe
  cf.br ^next
^next:
  test.greedy_cfg_rewrite observe
  return
}

// -----

// Erasing a labeled block makes its successors unreachable, even one that
// keeps a predecessor, here itself.
// CHECK-LABEL: func.func @erase_label_block
// CHECK:         test.label "W"
// CHECK-NEXT:    cf.br ^bb2
// CHECK-NEXT:  ^bb2:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
// OFF-NEXT:    ^bb3:
// OFF-NEXT:      return
// CHECK-NEXT:  }
func.func @erase_label_block(%c: i1) {
  cf.br ^start
^warm:
  test.label "W"
  cf.br ^warm2
^warm2:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
^start:
  test.greedy_cfg_rewrite erase_next
  return
^label:
  test.label "L"
  cf.br ^loop
^loop:
  test.greedy_cfg_rewrite observe
  cf.cond_br %c, ^loop, ^exit
^exit:
  test.greedy_cfg_rewrite observe
  return
}

// -----

// Moving a labeled block out of a region op makes the block holding that op
// unreachable, even though it keeps a predecessor, here itself. The observer
// in ^warm queries the cache, which then holds both holding blocks, before
// the move.
// CHECK-LABEL: func.func @move_label
// CHECK:         test.label "M"
// CHECK-NEXT:    "test.return"() : () -> ()
// CHECK-NEXT:  }) {label_target} : () -> ()
// CHECK-NEXT:  return
// CHECK-NEXT: }
func.func @move_label(%c: i1) {
  cf.br ^start
^warm:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "W"
    "test.return"() : () -> ()
  }) : () -> ()
  cf.cond_br %c, ^warm, ^start
^start:
  test.greedy_cfg_rewrite move_label
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  }) {label_target} : () -> ()
  return
^source:
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "M"
    "test.return"() : () -> ()
  }) {label_source} : () -> ()
  cf.cond_br %c, ^source, ^start
}

// -----

// Replacing a label inserts the new op in front of the label before erasing
// it. The block and the blocks behind it become unreachable.
// CHECK-LABEL: func.func @replace_label
// CHECK:         test.label "W"
// CHECK-NEXT:    cf.br ^bb2
// CHECK-NEXT:  ^bb2:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
// OFF-NEXT:    ^bb3:
// OFF-NEXT:      return
// CHECK-NEXT:  }
func.func @replace_label() {
  cf.br ^start
^warm:
  test.label "W"
  cf.br ^warm2
^warm2:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
^start:
  test.greedy_cfg_rewrite replace_next_label
  return
^label:
  test.label "L"
  cf.br ^next
^next:
  test.greedy_cfg_rewrite observe
  return
}

// -----

// A labeled block that loses its last CFG edge is still entered through its
// label, and so are the blocks behind it.
// CHECK-LABEL: func.func @label_loses_edge
// CHECK:         test.label "L"
// CHECK-NEXT:    cf.br ^[[NEXT:bb[0-9]+]]
// CHECK-NEXT:  ^[[NEXT]]:
// OFF-NEXT:      test.greedy_cfg_rewrite observe
// CHECK-NEXT:    return
func.func @label_loses_edge(%c: i1) {
  cf.cond_br %c, ^start, ^exit
^start:
  test.greedy_cfg_rewrite redirect
  cf.br ^label
^label:
  test.label "L"
  cf.br ^next
^next:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
^exit:
  return
}

// -----

// Moving an op that holds a label makes the block it leaves unreachable, even
// though that block keeps a predecessor, here itself. The observer in ^warm
// queries the cache, which then holds both holding blocks, before the move.
// CHECK-LABEL: func.func @move_holder
// CHECK:         "test.ssacfg_region"() ({
// CHECK:           test.label "M"
// CHECK:         }) {label_source} : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @move_holder(%c: i1) {
  cf.br ^start
^warm:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "W"
    "test.return"() : () -> ()
  }) : () -> ()
  cf.cond_br %c, ^warm, ^start
^start:
  test.greedy_cfg_rewrite move_holder
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe {label_target}
  return
^source:
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "M"
    "test.return"() : () -> ()
  }) {label_source} : () -> ()
  test.greedy_cfg_rewrite observe
  cf.cond_br %c, ^source, ^start
}

// -----

// A labeled block that loses its last CFG edge stays in the reachability
// cache: when it later branches to a new block, that block is reachable too.
// The observer in ^warm2 makes the cache search for labels first, so the new
// block is not found by searching again.
// CHECK-LABEL: func.func @label_loses_edge_then_branches
// CHECK:         test.label "L"
// OFF-NEXT:      test.greedy_cfg_rewrite append_reachable %{{.*}}
// OFF-NEXT:      cf.br
// ON-NEXT:       cf.cond_br
func.func @label_loses_edge_then_branches(%c: i1) {
  cf.br ^start
^warm:
  test.label "W"
  cf.br ^warm2
^warm2:
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
^start:
  test.greedy_cfg_rewrite redirect
  cf.br ^label
^label:
  test.label "L"
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite append_reachable %c
  cf.br ^exit
^exit:
  return
}
