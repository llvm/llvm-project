// RUN: mlir-opt %s -split-input-file -test-strict-pattern-driver="rewrite-after-labels" -verify-diagnostics | FileCheck %s

// The op-based driver does not erase unreachable blocks. With label
// reachability, it processes the blocks that control enters through a label
// and still skips the others.
// CHECK-LABEL: func.func @label_block
// CHECK:         test.label "L"
// CHECK-NEXT:    cf.br ^[[NEXT:bb[0-9]+]]
// CHECK-NEXT:  ^[[NEXT]]:
// CHECK-NEXT:    return
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    test.greedy_cfg_rewrite observe
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

// A block that holds a nested label and whose only predecessor is unreachable
// is entered through the label, so its operations before the holder op stay
// unreachable.
// CHECK-LABEL: func.func @holder_with_unreachable_pred
// CHECK:       ^bb2:
// CHECK-NEXT:    test.greedy_cfg_rewrite observe
// CHECK-NEXT:    "test.ssacfg_region"() ({
// CHECK-NEXT:      "test.return"() : () -> ()
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      test.label "L"
// CHECK-NEXT:      "test.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    return
func.func @holder_with_unreachable_pred() {
  return
^pred:
  cf.br ^holder
^holder:
  test.greedy_cfg_rewrite observe
  "test.ssacfg_region"() ({
    "test.return"() : () -> ()
  ^label:
    test.label "L"
    // expected-remark @+1 {{processed reachable block}}
    test.greedy_cfg_rewrite observe
    "test.return"() : () -> ()
  }) : () -> ()
  // expected-remark @+1 {{processed reachable block}}
  test.greedy_cfg_rewrite observe
  return
}
