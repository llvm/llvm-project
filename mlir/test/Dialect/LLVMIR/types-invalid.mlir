// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @array_of_void() {
  // expected-error @+1 {{invalid array element type}}
  "test.some_op"() : () -> !llvm.array<4 x void>
}

// -----

func.func @function_returning_function() {
  // expected-error @+1 {{invalid function result type}}
  "test.some_op"() : () -> !llvm.func<func<void ()> ()>
}

// -----

func.func @function_taking_opaque_struct() {
  // expected-error @+1 {{invalid function argument type}}
  "test.some_op"() : () -> !llvm.func<void(struct<"foo", opaque>)>
}

// -----

func.func @function_taking_function() {
  // expected-error @+1 {{invalid function argument type}}
  "test.some_op"() : () -> !llvm.func<void (func<void ()>)>
}

// -----

func.func @repeated_struct_name() {
  "test.some_op"() : () -> !llvm.struct<"a", (ptr)>
  // expected-error @+1 {{identified type already used with a different body}}
  "test.some_op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func.func @repeated_struct_name_packed() {
  "test.some_op"() : () -> !llvm.struct<"a", packed (i32)>
  // expected-error @+1 {{identified type already used with a different body}}
  "test.some_op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func.func @repeated_struct_opaque() {
  "test.some_op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "test.some_op"() : () -> !llvm.struct<"a", ()>
}

// -----

func.func @repeated_struct_opaque_non_empty() {
  "test.some_op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "test.some_op"() : () -> !llvm.struct<"a", (i32, i32)>
}

// -----

func.func @repeated_struct_opaque_redefinition() {
  "test.some_op"() : () -> !llvm.struct<"a", ()>
  // expected-error @+1 {{redeclaring defined struct as opaque}}
  "test.some_op"() : () -> !llvm.struct<"a", opaque>
}

// -----

func.func @struct_literal_opaque() {
  // expected-error @+1 {{only identified structs can be opaque}}
  "test.some_op"() : () -> !llvm.struct<opaque>
}

// -----

func.func @top_level_struct_no_body() {
  // expected-error @below {{struct without a body only allowed in a recursive struct}}
  "test.some_op"() : () -> !llvm.struct<"a">
}

// -----

func.func @nested_redefine_attempt() {
  // expected-error @below {{identifier already used for an enclosing struct}}
  "test.some_op"() : () -> !llvm.struct<"a", (struct<"a", ()>)>
}

// -----

func.func @unexpected_type() {
  // expected-error @+1 {{unexpected type, expected keyword}}
  "test.some_op"() : () -> !llvm.tensor<*xf32>
}

// -----

func.func @unexpected_type() {
  // expected-error @+1 {{unknown LLVM type}}
  "test.some_op"() : () -> !llvm.ifoo
}

// -----

func.func @invalid_di_derived_type_extra_data() {
  // expected-error @+1 {{extraData must be a DINodeAttr or an IntegerAttr}}
  "test.some_op"() {attr = #llvm.di_derived_type<tag = DW_TAG_member, sizeInBits = 64, extraData = "not debug info">} : () -> ()
}

// -----

func.func @explicitly_opaque_struct() {
  "test.some_op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "test.some_op"() : () -> !llvm.struct<"a", ()>
}

// -----

func.func @literal_struct_with_void() {
  // expected-error @+1 {{invalid LLVM structure element type}}
  "test.some_op"() : () -> !llvm.struct<(void)>
}

// -----

func.func @identified_struct_with_void() {
  // expected-error @+1 {{invalid LLVM structure element type}}
  "test.some_op"() : () -> !llvm.struct<"a", (void)>
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
  "test.some_op"() : () -> !llvm.target<"target1", 5, i32, 1>
}

// -----

func.func private @target_ext_no_name() {
  // expected-error@below {{expected string}}
  // expected-error@below {{failed to parse LLVMTargetExtType parameter 'extTypeName' which is to be a `::llvm::StringRef`}}
  "test.some_op"() : () -> !llvm.target<i32, 42>
}

// -----

func.func @byte_invalid_bitwidth() {
    // expected-error@below {{bitwidth must be less than 8388608, but got 8388608}}
    %0 = "test.some_op"() : () -> !llvm.byte<8388608>
    llvm.return
}

// -----

llvm.func @byte_zero_bitwidth() {
    // expected-error@below {{bitwidth must be greater than 0}}
    %0 = "test.some_op"() : () -> !llvm.byte<0>
    llvm.return
}
