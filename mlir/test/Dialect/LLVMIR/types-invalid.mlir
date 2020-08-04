// RUN: mlir-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s

func @repeated_struct_name() {
  "some.op"() : () -> !llvm2.struct<"a", (ptr<struct<"a">>)>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: (ptr<struct<"a">>)}}
  "some.op"() : () -> !llvm2.struct<"a", (i32)>
}

// -----

func @repeated_struct_name_packed() {
  "some.op"() : () -> !llvm2.struct<"a", packed (i32)>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: packed (i32)}}
  "some.op"() : () -> !llvm2.struct<"a", (i32)>
}

// -----

func @repeated_struct_opaque() {
  "some.op"() : () -> !llvm2.struct<"a", opaque>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: opaque}}
  "some.op"() : () -> !llvm2.struct<"a", ()>
}

// -----

func @repeated_struct_opaque_non_empty() {
  "some.op"() : () -> !llvm2.struct<"a", opaque>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: opaque}}
  "some.op"() : () -> !llvm2.struct<"a", (i32, i32)>
}

// -----

func @repeated_struct_opaque_redefinition() {
  "some.op"() : () -> !llvm2.struct<"a", ()>
  // expected-error @+1 {{redeclaring defined struct as opaque}}
  "some.op"() : () -> !llvm2.struct<"a", opaque>
}

// -----

func @struct_literal_opaque() {
  // expected-error @+1 {{only identified structs can be opaque}}
  "some.op"() : () -> !llvm2.struct<opaque>
}

// -----

func @unexpected_type() {
  // expected-error @+1 {{unexpected type, expected i* or keyword}}
  "some.op"() : () -> !llvm2.f32
}

// -----

func @unexpected_type() {
  // expected-error @+1 {{unknown LLVM type}}
  "some.op"() : () -> !llvm2.ifoo
}

// -----

func @explicitly_opaque_struct() {
  "some.op"() : () -> !llvm2.struct<"a", opaque>
  // expected-error @+2 {{identified type already used with a different body}}
  // expected-note @+1 {{existing body: opaque}}
  "some.op"() : () -> !llvm2.struct<"a", ()>
}

// -----

func @dynamic_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm2.vec<? x float>
}

// -----

func @dynamic_scalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm2.vec<? x ? x float>
}

// -----

func @unscalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm2.vec<4 x 4 x i32>
}

