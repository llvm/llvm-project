// RUN: mlir-opt %s -split-input-file -verify-diagnostics| FileCheck %s

// CHECK-LABEL: @test_enum_attr_roundtrip
func.func @test_enum_attr_roundtrip() -> () {
  // CHECK: attr = #test.custom_float<"float" : 2.000000e+00>
  "test.op"() {attr = #test.custom_float<"float" : 2.>} : () -> ()
  // CHECK: attr = #test.custom_float<"double" : 2.000000e+00>
  "test.op"() {attr = #test.custom_float<"double" : 2.>} : () -> ()
   // CHECK: attr = #test.custom_float<"fp80" : 2.000000e+00>
  "test.op"() {attr = #test.custom_float<"fp80" : 2.>} : () -> ()
  // CHECK: attr = #test.custom_float<"float" : 0x7FC00000>
  "test.op"() {attr = #test.custom_float<"float" : 0x7FC00000>} : () -> ()
  // CHECK: attr = #test.custom_float<"double" : 0x7FF0000001000000>
  "test.op"() {attr = #test.custom_float<"double" : 0x7FF0000001000000>} : () -> ()
  // CHECK: attr = #test.custom_float<"fp80" : 0x7FFFC000000000100000>
  "test.op"() {attr = #test.custom_float<"fp80" : 0x7FFFC000000000100000>} : () -> ()
  return
}

// -----

// Verify literal must be hex or float

// expected-error @below {{unexpected decimal integer literal for a floating point value}}
// expected-note @below {{add a trailing dot to make the literal a float}}
"test.op"() {attr = #test.custom_float<"float" : 42>} : () -> ()

// -----

// Integer value must be in the width of the floating point type

// expected-error @below {{hexadecimal float constant out of range for type}}
"test.op"() {attr = #test.custom_float<"float" : 0x7FC000000>} : () -> ()


// -----

// Integer value must be in the width of the floating point type

// expected-error @below {{hexadecimal float constant out of range for type}}
"test.op"() {attr = #test.custom_float<"double" : 0x7FC000007FC0000000>} : () -> ()


// -----

// Integer value must be in the width of the floating point type

// expected-error @below {{hexadecimal float constant out of range for type}}
"test.op"() {attr = #test.custom_float<"fp80" : 0x7FC0000007FC0000007FC000000>} : () -> ()

// -----

// Value must be a floating point literal or integer literal

// expected-error @below {{expected floating point literal}}
"test.op"() {attr = #test.custom_float<"float" : "blabla">} : () -> ()

