// RUN: mlir-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s

func.func @array_of_void() {
  // expected-error @+1 {{invalid array element type}}
  "some.op"() : () -> !llvm.array<4 x void>
}

// -----

func.func @function_returning_function() {
  // expected-error @+1 {{invalid function result type}}
  "some.op"() : () -> !llvm.func<func<void ()> ()>
}

// -----

func.func @function_taking_function() {
  // expected-error @+1 {{invalid function argument type}}
  "some.op"() : () -> !llvm.func<void (func<void ()>)>
}

// -----

func.func @repeated_struct_name() {
  "some.op"() : () -> !llvm.struct<"a", (ptr)>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func.func @repeated_struct_name_packed() {
  "some.op"() : () -> !llvm.struct<"a", packed (i32)>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func.func @repeated_struct_opaque() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", ()>
}

// -----

func.func @repeated_struct_opaque_non_empty() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", (i32, i32)>
}

// -----

func.func @repeated_struct_opaque_redefinition() {
  "some.op"() : () -> !llvm.struct<"a", ()>
  // expected-error @+1 {{redeclaring defined struct as opaque}}
  "some.op"() : () -> !llvm.struct<"a", opaque>
}

// -----

func.func @struct_literal_opaque() {
  // expected-error @+1 {{only identified structs can be opaque}}
  "some.op"() : () -> !llvm.struct<opaque>
}

// -----

func.func @top_level_struct_no_body() {
  // expected-error @below {{struct without a body only allowed in a recursive struct}}
  "some.op"() : () -> !llvm.struct<"a">
}

// -----

func.func @nested_redefine_attempt() {
  // expected-error @below {{identifier already used for an enclosing struct}}
  "some.op"() : () -> !llvm.struct<"a", (struct<"a", ()>)>
}

// -----

func.func @unexpected_type() {
  // expected-error @+1 {{unexpected type, expected keyword}}
  "some.op"() : () -> !llvm.tensor<*xf32>
}

// -----

func.func @unexpected_type() {
  // expected-error @+1 {{unknown LLVM type}}
  "some.op"() : () -> !llvm.ifoo
}

// -----

func.func @explicitly_opaque_struct() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", ()>
}

// -----

func.func @literal_struct_with_void() {
  // expected-error @+1 {{invalid LLVM structure element type}}
  "some.op"() : () -> !llvm.struct<(void)>
}

// -----

func.func @identified_struct_with_void() {
  // expected-error @+1 {{invalid LLVM structure element type}}
  "some.op"() : () -> !llvm.struct<"a", (void)>
}

// -----

// expected-error @+1 {{unexpected type, expected keyword}}
func.func private @unexpected_type() -> !llvm.tensor<*xf32>

// -----

// expected-error @+1 {{unexpected type, expected keyword}}
func.func private @unexpected_type() -> !llvm.f32

// -----

func.func private @target_ext_invalid_order() {
  // expected-error @+1 {{failed to parse parameter list for target extension type}}
  "some.op"() : () -> !llvm.target<"target1", 5, i32, 1>
}

// -----

func.func private @target_ext_no_name() {
  // expected-error@below {{expected string}}
  // expected-error@below {{failed to parse LLVMTargetExtType parameter 'extTypeName' which is to be a `::llvm::StringRef`}}
  "some.op"() : () -> !llvm.target<i32, 42>
}
