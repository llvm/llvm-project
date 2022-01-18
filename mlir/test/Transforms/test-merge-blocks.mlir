// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-merge-blocks -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @merge_blocks
func @merge_blocks(%arg0: i32, %arg1 : i32) -> () {
  //      CHECK: "test.merge_blocks"() ({
  // CHECK-NEXT:   "test.return"
  // CHECK-NEXT: })
  // CHECK-NEXT: "test.return"
  %0:2 = "test.merge_blocks"() ({
  ^bb0:
     "test.br"(%arg0, %arg1)[^bb1] : (i32, i32) -> ()
  ^bb1(%arg3 : i32, %arg4 : i32):
     "test.return"(%arg3, %arg4) : (i32, i32) -> ()
  }) : () -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}

// -----

// The op in this function is rewritten to itself (and thus remains
// illegal) by a pattern that merges the second block with the first
// after adding an operation into it.  Check that we can undo block
// removal successfully.
// CHECK-LABEL: @undo_blocks_merge
func @undo_blocks_merge(%arg0: i32) {
  "test.undo_blocks_merge"() ({
    // expected-remark@-1 {{op 'test.undo_blocks_merge' is not legalizable}}
    // CHECK: "unregistered.return"(%{{.*}})[^[[BB:.*]]]
    "unregistered.return"(%arg0)[^bb1] : (i32) -> ()
    // expected-remark@-1 {{op 'unregistered.return' is not legalizable}}
  // CHECK: ^[[BB]]
  ^bb1(%arg1 : i32):
    // CHECK: "unregistered.return"
    "unregistered.return"(%arg1) : (i32) -> ()
    // expected-remark@-1 {{op 'unregistered.return' is not legalizable}}
  }) : () -> ()
}

// -----

// CHECK-LABEL: @inline_regions()
func @inline_regions() -> ()
{
  //      CHECK: test.SingleBlockImplicitTerminator
  // CHECK-NEXT:   %[[T0:.*]] = "test.type_producer"
  // CHECK-NEXT:   "test.type_consumer"(%[[T0]])
  // CHECK-NEXT:   "test.finish"
  "test.SingleBlockImplicitTerminator"() ({
  ^bb0:
    %0 = "test.type_producer"() : () -> i32
    "test.SingleBlockImplicitTerminator"() ({
    ^bb1:
      "test.type_consumer"(%0) : (i32) -> ()
      "test.finish"() : () -> ()
    }) : () -> ()
    "test.finish"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}
