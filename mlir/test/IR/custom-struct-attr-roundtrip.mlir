// RUN: mlir-opt %s -split-input-file -verify-diagnostics| FileCheck %s

// CHECK-LABEL: @test_struct_attr_roundtrip
func.func @test_struct_attr_roundtrip() -> () {
  // CHECK: attr = #test.custom_struct<type_str = "struct", value = 2, opt_value = [3, 3]>
  "test.op"() {attr = #test.custom_struct<type_str = "struct", value = 2, opt_value = [3, 3]>} : () -> ()
  // CHECK: attr = #test.custom_struct<type_str = "struct", value = ?, opt_value = 1>
  "test.op"() {attr = #test.custom_struct<type_str = "struct", value = ?, opt_value = 1>} : () -> ()
  // CHECK: attr = #test.custom_struct<type_str = "struct", value = 2, opt_value = [3, 3]>
  "test.op"() {attr = #test.custom_struct<value = 2, type_str = "struct", opt_value = [3, 3]>} : () -> ()
  // CHECK: attr = #test.custom_struct<type_str = "struct", value = 2>
  "test.op"() {attr = #test.custom_struct<type_str = "struct", value = 2>} : () -> ()
  // CHECK: attr = #test.custom_struct<type_str = "struct", value = 2>
  "test.op"() {attr = #test.custom_struct<value = 2, type_str = "struct">} : () -> ()
  return
}

// -----

// Verify all required parameters must be provided. `value` is missing.

// expected-error @below {{struct is missing required parameter: value}}
"test.op"() {attr = #test.custom_struct<type_str = "struct">} : () -> ()

// -----

// Verify all keywords must be provided. All missing.

// expected-error @below {{expected valid keyword}}
// expected-error @below {{expected a parameter name in struct}}
"test.op"() {attr = #test.custom_struct<"struct", 2>} : () -> ()

// -----

// Verify all keywords must be provided. `type_str` missing.

// expected-error @below {{expected valid keyword}}
// expected-error @below {{expected a parameter name in struct}}
"test.op"() {attr = #test.custom_struct<"struct", value = 2, opt_value = [3, 3]>} : () -> ()

// -----

// Verify all keywords must be provided. `value` missing.

// expected-error @below {{expected valid keyword}}
// expected-error @below {{expected a parameter name in struct}}
"test.op"() {attr = #test.custom_struct<type_str = "struct", 2>} : () -> ()

// -----

// Verify invalid keyword provided.

// expected-error @below {{duplicate or unknown struct parameter name: type_str2}}
"test.op"() {attr = #test.custom_struct<type_str2 = "struct", value = 2>} : () -> ()

// -----

// Verify duplicated keyword provided.

// expected-error @below {{duplicate or unknown struct parameter name: type_str}}
"test.op"() {attr = #test.custom_struct<type_str = "struct", type_str = "struct2", value = 2>} : () -> ()

// -----

// Verify equals missing.

// expected-error @below {{expected '='}}
"test.op"() {attr = #test.custom_struct<type_str "struct", value = 2>} : () -> ()
