// RUN: mlir-opt %s -test-legalize-type-conversion -allow-unregistered-dialect -split-input-file -verify-diagnostics | FileCheck %s


func @test_invalid_arg_materialization(
  // expected-error@below {{failed to materialize conversion for block argument #0 that remained live after conversion, type was 'i16'}}
  %arg0: i16) {
  // expected-note@below {{see existing live user here}}
  "foo.return"(%arg0) : (i16) -> ()
}

// -----

// CHECK-LABEL: func @test_valid_arg_materialization
func @test_valid_arg_materialization(%arg0: i64) {
  // CHECK: %[[ARG:.*]] = "test.type_producer"
  // CHECK: "foo.return"(%[[ARG]]) : (i64)

  "foo.return"(%arg0) : (i64) -> ()
}

// -----

func @test_invalid_result_materialization() {
  // expected-error@below {{failed to materialize conversion for result #0 of operation 'test.type_producer' that remained live after conversion}}
  %result = "test.type_producer"() : () -> f16

  // expected-note@below {{see existing live user here}}
  "foo.return"(%result) : (f16) -> ()
}

// -----

func @test_invalid_result_materialization() {
  // expected-error@below {{failed to materialize conversion for result #0 of operation 'test.type_producer' that remained live after conversion}}
  %result = "test.type_producer"() : () -> f16

  // expected-note@below {{see existing live user here}}
  "foo.return"(%result) : (f16) -> ()
}

// -----

// CHECK-LABEL: @test_transitive_use_materialization
func @test_transitive_use_materialization() {
  // CHECK: %[[V:.*]] = "test.type_producer"() : () -> f64
  // CHECK: %[[C:.*]] = "test.cast"(%[[V]]) : (f64) -> f32
  %result = "test.another_type_producer"() : () -> f32
  // CHECK: "foo.return"(%[[C]])
  "foo.return"(%result) : (f32) -> ()
}

// -----

func @test_transitive_use_invalid_materialization() {
  // expected-error@below {{failed to materialize conversion for result #0 of operation 'test.type_producer' that remained live after conversion}}
  %result = "test.another_type_producer"() : () -> f16
  // expected-note@below {{see existing live user here}}
  "foo.return"(%result) : (f16) -> ()
}

// -----

// CHECK-LABEL: func @test_valid_result_legalization
func @test_valid_result_legalization() {
  // CHECK: %[[RESULT:.*]] = "test.type_producer"() : () -> f64
  // CHECK: %[[CAST:.*]] = "test.cast"(%[[RESULT]]) : (f64) -> f32
  // CHECK: "foo.return"(%[[CAST]]) : (f32)

  %result = "test.type_producer"() : () -> f32
  "foo.return"(%result) : (f32) -> ()
}

// -----

// Should not segfault here but gracefully fail.
// CHECK-LABEL: func @test_signature_conversion_undo
func @test_signature_conversion_undo() {
  // CHECK: test.signature_conversion_undo
  "test.signature_conversion_undo"() ({
  // CHECK: ^{{.*}}(%{{.*}}: f32):
  ^bb0(%arg0: f32):
    "test.type_consumer"(%arg0) : (f32) -> ()
    "test.return"(%arg0) : (f32) -> ()
  }) : () -> ()
  return
}

// -----

// Should not segfault here but gracefully fail.
// CHECK-LABEL: func @test_block_argument_not_converted
func @test_block_argument_not_converted() {
  "test.unsupported_block_arg_type"() ({
    // NOTE: The test pass does not convert `index` types.
    // CHECK: ^bb0({{.*}}: index):
    ^bb0(%0 : index):
      "test.return"(%0) : (index) -> ()
  }) : () -> ()
  return
}
