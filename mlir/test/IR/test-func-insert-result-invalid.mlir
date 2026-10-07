// RUN: mlir-opt %s -test-func-insert-result -split-input-file -verify-diagnostics

// expected-error @below {{failed to insert results}}
func.func private @f() attributes {test.insert_results = [
  [0, f32, {test.A}],
  [1, f32, {test.B}]]}

// -----

// expected-error @below {{failed to insert results}}
func.func private @f() -> (f32 {test.A}) attributes {test.insert_results = [
  [1, f32, {test.B}],
  [0, f32, {test.C}]]}
