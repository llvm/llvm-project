// RUN: mlir-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s

func.func @void_pointer() {
  // expected-error @+1 {{invalid pointer element type}}
  "some.op"() : () -> !llvm.ptr<void>
}

// -----

func.func @repeated_struct_name() {
  "some.op"() : () -> !llvm.struct<"a", (ptr<struct<"a">>)>
  // expected-error @+1 {{identified type already used with a different body}}
  "some.op"() : () -> !llvm.struct<"a", (i32)>
}

// -----

func.func @dynamic_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<? x ptr<f32>>
}

// -----

func.func @dynamic_scalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<?x? x ptr<f32>>
}

// -----

func.func @unscalable_vector() {
  // expected-error @+1 {{expected '? x <integer> x <type>' or '<integer> x <type>'}}
  "some.op"() : () -> !llvm.vec<4x4 x ptr<i32>>
}

// -----

func.func @zero_vector() {
  // expected-error @+1 {{the number of vector elements must be positive}}
  "some.op"() : () -> !llvm.vec<0 x ptr<i32>>
}
