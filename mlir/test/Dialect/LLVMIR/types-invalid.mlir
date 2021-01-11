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
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func @repeated_struct_name_packed() {
  "some.op"() : () -> !llvm.struct<"a", packed (i32)>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func @repeated_struct_opaque() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", ()>
}

// -----

func @repeated_struct_opaque_non_empty() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
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
  // expected-error @+1 {{unexpected type, expected keyword}}
  "some.op"() : () -> !llvm.tensor<*xf32>
}

// -----

func @unexpected_type() {
  // expected-error @+1 {{unknown LLVM type}}
  "some.op"() : () -> !llvm.ifoo
}

// -----

func @explicitly_opaque_struct() {
  "some.op"() : () -> !llvm.struct<"a", opaque>
  // expected-error @+1 {{identified type already used with a different body}}
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
  "some.op"() : () -> !llvm.vec<? x ptr<f32>>
}

// -----

func @dynamic_scalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<?x? x ptr<f32>>
}

// -----

func @unscalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<4x4 x ptr<i32>>
}

// -----

func @zero_vector() {
  // expected-error @+1 {{the number of vector elements must be positive}}
  "some.op"() : () -> !llvm.vec<0 x ptr<i32>>
}

// -----

func @nested_vector() {
  // expected-error @+1 {{invalid vector element type}}
  "some.op"() : () -> !llvm.vec<2 x vector<2xi32>>
}

// -----

func @scalable_void_vector() {
  // expected-error @+1 {{invalid vector element type}}
  "some.op"() : () -> !llvm.vec<?x4 x void>
}

// -----

// expected-warning @+1 {{deprecated syntax, drop '!llvm.' for integers}}
func private @deprecated_int() -> !llvm.i32

// -----

// expected-error @+1 {{unexpected type, expected keyword}}
func private @unexpected_type() -> !llvm.tensor<*xf32>

// -----

// expected-warning @+1 {{deprecated syntax, use bf16 instead}}
func private @deprecated_bfloat() -> !llvm.bfloat

// -----

// expected-warning @+1 {{deprecated syntax, use f16 instead}}
func private @deprecated_half() -> !llvm.half

// -----

// expected-warning @+1 {{deprecated syntax, use f32 instead}}
func private @deprecated_float() -> !llvm.float

// -----

// expected-warning @+1 {{deprecated syntax, use f64 instead}}
func private @deprecated_double() -> !llvm.double

// -----

// expected-error @+1 {{unexpected type, expected keyword}}
func private @unexpected_type() -> !llvm.f32
