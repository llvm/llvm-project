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

func.func @dynamic_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<? x ptr>
}

// -----

func.func @dynamic_scalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<?x? x ptr>
}

// -----

func.func @unscalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<4x4 x ptr>
}

// -----

func.func @zero_vector() {
  // expected-error @+1 {{the number of vector elements must be positive}}
  "some.op"() : () -> !llvm.vec<0 x ptr>
}

// -----

func.func @nested_vector() {
  // expected-error @+1 {{invalid vector element type}}
  "some.op"() : () -> !llvm.vec<2 x vector<2xi32>>
}

// -----

func.func @scalable_void_vector() {
  // expected-error @+1 {{invalid vector element type}}
  "some.op"() : () -> !llvm.vec<?x4 x void>
}

// -----

// expected-error @+1 {{unexpected type, expected keyword}}
func.func private @unexpected_type() -> !llvm.tensor<*xf32>

// -----

// expected-error @+1 {{unexpected type, expected keyword}}
func.func private @unexpected_type() -> !llvm.f32

// -----

// expected-error @below {{cannot use !llvm.vec for built-in primitives, use 'vector' instead}}
func.func private @llvm_vector_primitive() -> !llvm.vec<4 x f32>

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
