// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -verify-diagnostics %s

ml_program.func @ssa_enforced(%arg0 : i32) -> i32 {
  // expected-error @+1 {{does not dominate this use}}
  %1 = "unregistered.dummy"(%0) : (i32) -> i32
  // expected-note @+1 {{operand defined here}}
  %0 = "unregistered.dummy"(%arg0) : (i32) -> i32
  ml_program.return %0 : i32
}

// -----
ml_program.func @return_arity_match(%arg0 : i32) -> i32 {
  // expected-error @+1 {{enclosing function (@return_arity_match) returns 1}}
  ml_program.return %arg0, %arg0 : i32, i32
}

// -----
ml_program.func @return_type_match(%arg0 : i64) -> i32 {
  // expected-error @+1 {{doesn't match function result}}
  ml_program.return %arg0 : i64
}

// -----
ml_program.subgraph @output_arity_match(%arg0 : i32) -> i32 {
  // expected-error @+1 {{enclosing function (@output_arity_match) outputs 1}}
  ml_program.output %arg0, %arg0 : i32, i32
}

// -----
ml_program.subgraph @output_type_match(%arg0 : i64) -> i32 {
  // expected-error @+1 {{doesn't match function result}}
  ml_program.output %arg0 : i64
}

// -----
// expected-error @+1 {{immutable global must have an initial value}}
ml_program.global private @const : i32

// -----
ml_program.func @undef_global() -> i32 {
  // expected-error @+1 {{undefined global: nothere}}
  %0 = ml_program.global_load_const @nothere : i32
  ml_program.return %0 : i32
}

// -----
ml_program.global private mutable @var : i32
ml_program.func @mutable_const_load() -> i32 {
  // expected-error @+1 {{op cannot load as const from mutable global var}}
  %0 = ml_program.global_load_const @var : i32
  ml_program.return %0 : i32
}

// -----
ml_program.global private @var(42 : i64) : i64
ml_program.func @const_load_type_mismatch() -> i32 {
  // expected-error @+1 {{cannot load from global typed 'i64' as 'i32'}}
  %0 = ml_program.global_load_const @var : i32
  ml_program.return %0 : i32
}
