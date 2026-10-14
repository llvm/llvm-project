// RUN: mlir-opt %s -split-input-file -verify-diagnostics

llvm.func @missing_value() {
  // expected-error@below {{expected one value for each immediate argument position}}
  llvm.call_intrinsic "llvm.ubsantrap"() {immarg_positions = array<i32: 0>} : () -> ()
  llvm.return
}

// -----

llvm.func @missing_position() {
  // expected-error@below {{expected one value for each immediate argument position}}
  llvm.call_intrinsic "llvm.ubsantrap"() {immarg_values = [7 : i8]} : () -> ()
  llvm.return
}

// -----

llvm.func @negative_position() {
  // expected-error@below {{immediate argument position -1 is out of range for 1 arguments}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: -1>, immarg_values = [7 : i8]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @out_of_range_position(%bundle_arg: i32) {
  // Bundle operands do not count towards the number of intrinsic arguments.
  // expected-error@below {{immediate argument position 1 is out of range for 1 arguments}}
  llvm.call_intrinsic "llvm.ubsantrap"() ["tag"(%bundle_arg : i32)] {
    immarg_positions = array<i32: 1>, immarg_values = [7 : i8]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @duplicate_positions() {
  // expected-error@below {{immediate argument positions must be strictly increasing}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 0, 0>, immarg_values = [7 : i8, 8 : i8]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @unsorted_positions() {
  // expected-error@below {{immediate argument positions must be strictly increasing}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 1, 0>, immarg_values = [7 : i8, 8 : i8]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @string_immediate() {
  // expected-error@below {{immediate argument value #0 must be an LLVM-compatible scalar integer or floating-point attribute}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 0>, immarg_values = ["invalid"]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @vector_immediate() {
  // expected-error@below {{immediate argument value #0 must be an LLVM-compatible scalar integer or floating-point attribute}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 0>,
    immarg_values = [dense<7> : vector<2xi8>]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @index_immediate() {
  // expected-error@below {{immediate argument value #0 must be an LLVM-compatible scalar integer or floating-point attribute}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 0>, immarg_values = [7 : index]
  } : () -> ()
  llvm.return
}

// -----

llvm.func @signed_immediate() {
  // expected-error@below {{immediate argument value #0 must be an LLVM-compatible scalar integer or floating-point attribute}}
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 0>, immarg_values = [7 : si8]
  } : () -> ()
  llvm.return
}
