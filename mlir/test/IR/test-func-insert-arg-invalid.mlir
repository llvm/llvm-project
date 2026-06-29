// RUN: mlir-opt %s -test-func-insert-arg -split-input-file -verify-diagnostics

// expected-error @below {{failed to insert arguments}}
func.func @f() attributes {test.insert_args = [
  [0, i1, {test.A}],
  [1, i2, {test.B}]]} {
  return
}

// -----

// expected-error @below {{failed to insert arguments}}
func.func @f(%arg0: i1 {test.A}) attributes {test.insert_args = [
  [1, i2, {test.B}],
  [0, i3, {test.C}]]} {
  return
}
