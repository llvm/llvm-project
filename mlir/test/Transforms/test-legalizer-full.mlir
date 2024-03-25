// RUN: mlir-opt -allow-unregistered-dialect -test-legalize-patterns -test-legalize-mode=full -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @multi_level_mapping
func.func @multi_level_mapping() {
  // CHECK: "test.type_producer"() : () -> f64
  // CHECK: "test.type_consumer"(%{{.*}}) : (f64) -> ()
  %result = "test.type_producer"() : () -> i32
  "test.type_consumer"(%result) : (i32) -> ()
  "test.return"() : () -> ()
}

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
// CHECK-LABEL: func @replace_non_root_illegal_op
func.func @replace_non_root_illegal_op() {
  // CHECK-NEXT: "test.legal_op_b"
  // CHECK-NEXT: test.return
  %result = "test.replace_non_root"() : () -> (i32)
  "test.return"() : () -> ()
}

// -----

// Test that children of recursively legal operations are ignored.
func.func @recursively_legal_invalid_op() {
  /// Operation that is statically legal.
  builtin.module attributes {test.recursively_legal} {
    %ignored = "test.illegal_op_f"() : () -> (i32)
  }
  /// Operation that is dynamically legal, i.e. the function has a pattern
  /// applied to legalize the argument type before it becomes recursively legal.
  builtin.module {
    func.func @dynamic_func(%arg: i64) attributes {test.recursively_legal} {
      %ignored = "test.illegal_op_f"() : () -> (i32)
      "test.return"() : () -> ()
    }
  }

  "test.return"() : () -> ()
}

// -----

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {

  // Test that region cloning can be properly undone.
  func.func @test_undo_region_clone() {
    "test.region"() ({
      ^bb1(%i0: i64):
        "test.invalid"(%i0) : (i64) -> ()
    }) {legalizer.should_clone} : () -> ()

    // expected-error@+1 {{failed to legalize operation 'test.illegal_op_f'}}
    %ignored = "test.illegal_op_f"() : () -> (i32)
    "test.return"() : () -> ()
  }

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

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {

  // Test that region inlining can be properly undone.
  func.func @test_undo_region_inline() {
    "test.region"() ({
      ^bb1(%i0: i64):
        // expected-error@+1 {{failed to legalize operation 'cf.br'}}
        cf.br ^bb2(%i0 : i64)
      ^bb2(%i1: i64):
        "test.invalid"(%i1) : (i64) -> ()
    }) {} : () -> ()

    "test.return"() : () -> ()
  }

}

// -----

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {

  // Test that multiple block erases can be properly undone.
  func.func @test_undo_block_erase() {
    // expected-error@+1 {{failed to legalize operation 'test.region'}}
    "test.region"() ({
      ^bb1(%i0: i64):
        cf.br ^bb3(%i0 : i64)
      ^bb2(%i1: i64):
        "test.invalid"(%i1) : (i64) -> ()
      ^bb3(%i2: i64):
        cf.br ^bb2(%i2 : i64)
    }) {legalizer.should_clone, legalizer.erase_old_blocks} : () -> ()

    "test.return"() : () -> ()
  }

}

// -----

// expected-remark@+1 {{applyFullConversion failed}}
builtin.module {

  func.func @create_unregistered_op_in_pattern() -> i32 {
    // expected-error@+1 {{failed to legalize operation 'test.illegal_op_g'}}
    %0 = "test.illegal_op_g"() : () -> (i32)
    "test.return"(%0) : (i32) -> ()
  }

}
