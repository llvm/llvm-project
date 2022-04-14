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
