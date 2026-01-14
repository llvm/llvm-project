// RUN: mlir-opt -allow-unregistered-dialect -test-legalize-patterns="test-legalize-mode=full" -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @multi_level_mapping
func.func @multi_level_mapping() {
  // CHECK: "test.type_producer"() : () -> f64
  // CHECK: "test.type_consumer"(%{{.*}}) : (f64) -> ()
  %result = "test.type_producer"() : () -> i32
  "test.type_consumer"(%result) : (i32) -> ()
  "test.return"() : () -> ()
}

// -----

// Test that operations that are erased don't need to be legalized.
// CHECK-LABEL: func @dropped_region_with_illegal_ops
func.func @dropped_region_with_illegal_ops() {
  // CHECK-NEXT: test.return
  "test.drop_region_op"() ({
    %ignored = "test.illegal_op_f"() : () -> (i32)
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @replace_non_root_illegal_op
func.func @replace_non_root_illegal_op() {
  // CHECK-NEXT: "test.legal_op_b"
  // CHECK-NEXT: test.return
  %result = "test.replace_non_root"() : () -> (i32)
  "test.return"() : () -> ()
}

// -----

// Test that children of recursively legal operations are ignored.

// CHECK-LABEL: func @recursively_legal_invalid_op
func.func @recursively_legal_invalid_op() {
  /// Operation that is statically legal.
  builtin.module attributes {test.recursively_legal} {
    // CHECK: "test.illegal_op_f"
    %ignored = "test.illegal_op_f"() : () -> (i32)
  }
  /// Operation that is dynamically legal, i.e. the function has a pattern
  /// applied to legalize the argument type before it becomes recursively legal.
  builtin.module {
    // CHECK: func @dynamic_func(%{{.*}}: f64)
    func.func @dynamic_func(%arg: i64) attributes {test.recursively_legal} {
      // CHECK: "test.illegal_op_f"
      %ignored = "test.illegal_op_f"() : () -> (i32)
      "test.return"() : () -> ()
    }
  }

  "test.return"() : () -> ()
}

// -----

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {

  // Test that unknown operations can be dynamically legal.
  func.func @test_unknown_dynamically_legal() {
    "foo.unknown_op"() {test.dynamically_legal} : () -> ()

    // expected-error@+1 {{failed to legalize operation 'foo.unknown_op'}}
    "foo.unknown_op"() {} : () -> ()
    "test.return"() : () -> ()
  }

}

// -----

// The region of "test.post_order_legalization" is converted before the op.

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {
func.func @test_preorder_legalization() {
  // expected-error@+1 {{failed to legalize operation 'test.post_order_legalization'}}
  "test.post_order_legalization"() ({
  ^bb0(%arg0: i64):
    // Not-explicitly-legal ops are not allowed to survive.
    "test.remaining_consumer"(%arg0) : (i64) -> ()
    "test.invalid"(%arg0) : (i64) -> ()
  }) : () -> ()
  return
}
}
