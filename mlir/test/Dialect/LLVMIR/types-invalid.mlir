// RUN: mlir-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s

func @array_of_void() {
  // expected-error @+1 {{invalid array element type}}
  "some.op"() : () -> !llvm.array<4 x void>
}

// -----

func @function_returning_function() {
  // expected-error @+1 {{invalid function result type}}
  "some.op"() : () -> !llvm.func<func<void ()> ()>
}

// -----

func @function_taking_function() {
  // expected-error @+1 {{invalid function argument type}}
  "some.op"() : () -> !llvm.func<void (func<void ()>)>
}

// -----

func @void_pointer() {
  // expected-error @+1 {{invalid pointer element type}}
  "some.op"() : () -> !llvm.ptr<void>
}

// -----

func @repeated_struct_name() {
  "some.op"() : () -> !llvm.struct<"a", (ptr<struct<"a">>)>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: (ptr<struct<"a">>)}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func @repeated_struct_name_packed() {
  "some.op"() : () -> !llvm.struct<"a", packed (i32)>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: packed (i32)}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func @repeated_struct_opaque() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: opaque}}
  "some.op"() : () -> !llvm.struct<"a", ()>
}

// -----

func @repeated_struct_opaque_non_empty() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: opaque}}
  "some.op"() : () -> !llvm.struct<"a", (i32, i32)>
}

// -----

func @repeated_struct_opaque_redefinition() {
  "some.op"() : () -> !llvm.struct<"a", ()>
  // expected-error @+1 {{redeclaring defined struct as opaque}}
  "some.op"() : () -> !llvm.struct<"a", opaque>
}

// -----

func @struct_literal_opaque() {
  // expected-error @+1 {{only identified structs can be opaque}}
  "some.op"() : () -> !llvm.struct<opaque>
}

// -----

func @unexpected_type() {
  // expected-error @+1 {{unexpected type, expected i* or keyword}}
  "some.op"() : () -> !llvm.f32
}

// -----

func @unexpected_type() {
  // expected-error @+1 {{unknown LLVM type}}
  "some.op"() : () -> !llvm.ifoo
}

// -----

func @explicitly_opaque_struct() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: opaque}}
  "some.op"() : () -> !llvm.struct<"a", ()>
}

// -----

func @literal_struct_with_void() {
  // expected-error @+1 {{invalid LLVM structure element type}}
  "some.op"() : () -> !llvm.struct<(void)>
}

// -----

func @identified_struct_with_void() {
  // expected-error @+1 {{invalid LLVM structure element type}}
  "some.op"() : () -> !llvm.struct<"a", (void)>
}

// -----

func @dynamic_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<? x float>
}

// -----

func @dynamic_scalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<? x ? x float>
}

// -----

func @unscalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<4 x 4 x i32>
}

// -----

func @zero_vector() {
  // expected-error @+1 {{the number of vector elements must be positive}}
  "some.op"() : () -> !llvm.vec<0 x i32>
}

// -----

func @nested_vector() {
  // expected-error @+1 {{invalid vector element type}}
  "some.op"() : () -> !llvm.vec<2 x vec<2 x i32>>
}

// -----

func @scalable_void_vector() {
  // expected-error @+1 {{invalid vector element type}}
  "some.op"() : () -> !llvm.vec<? x 4 x void>
}
