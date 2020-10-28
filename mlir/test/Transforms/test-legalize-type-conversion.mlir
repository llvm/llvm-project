// RUN: mlir-opt %s -test-legalize-type-conversion -allow-unregistered-dialect -split-input-file -verify-diagnostics | FileCheck %s

// expected-error@below {{failed to materialize conversion for block argument #0 that remained live after conversion, type was 'i16'}}
func @test_invalid_arg_materialization(%arg0: i16) {
  // expected-note@below {{see existing live user here}}
  "foo.return"(%arg0) : (i16) -> ()
}

// -----

// expected-error@below {{failed to legalize conversion operation generated for block argument}}
func @test_invalid_arg_illegal_materialization(%arg0: i32) {
  "foo.return"(%arg0) : (i32) -> ()
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

func @test_invalid_result_legalization() {
  // expected-error@below {{failed to legalize conversion operation generated for result #0 of operation 'test.type_producer' that remained live after conversion}}
  %result = "test.type_producer"() : () -> i16
  "foo.return"(%result) : (i16) -> ()
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
