// RUN: mlir-opt %s -test-named-attrs -split-input-file --verify-diagnostics | FileCheck %s

func.func @f_unit_attr() attributes {test.test_unit} { // expected-remark {{found unit attr}}
  %0:2 = "test.producer"() : () -> (i32, i32)
  return
}

// -----

func.func @f_unit_attr_fail() attributes {test.test_unit_fail} { // expected-error {{missing unit attr}}
  %0:2 = "test.producer"() : () -> (i32, i32)
  return
}

// -----

func.func @f_int_attr() attributes {test.test_int = 42 : i32} { // expected-remark {{correct int value}}
  %0:2 = "test.producer"() : () -> (i32, i32)
  return
}

// -----

func.func @f_int_attr_fail() attributes {test.test_int = 24 : i32} { // expected-error {{wrong int value}}
  %0:2 = "test.producer"() : () -> (i32, i32)
  return
}

// -----

func.func @f_int_attr_fail2() attributes {test.test_int_fail = 42 : i64} { // expected-error {{missing int attr}}
  %0:2 = "test.producer"() : () -> (i32, i32)
  return
}

// -----

func.func @f_lookup_attr() attributes {test.test_int = 42 : i64} { // expected-remark {{lookup found attr}}
  %0:2 = "test.producer"() : () -> (i32, i32) // expected-remark {{lookup found attr}}
  return // expected-remark {{lookup found attr}}
}

// -----

func.func @f_lookup_attr2() { // expected-error {{lookup failed}}
  "test.any_attr_of_i32_str"() {attr = 3 : i32, test.test_int = 24 : i32} : () -> () // expected-remark {{lookup found attr}}
  return // expected-error {{lookup failed}}
}

// -----

func.func @f_lookup_attr_fail() attributes {test.test_int_fail = 42 : i64} { // expected-error {{lookup failed}}
  %0:2 = "test.producer"() : () -> (i32, i32) // expected-error {{lookup failed}}
  return // expected-error {{lookup failed}}
}

// -----

// CHECK: func.func @f_set_attr() attributes {test.test_int = 42 : i32}
func.func @f_set_attr() { // expected-remark {{set int attr}}
  %0:2 = "test.producer"() : () -> (i32, i32)
  return
}

