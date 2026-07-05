// RUN: mlir-opt %s --irdl-file=%S/cyclic-types.irdl.mlir -split-input-file -verify-diagnostics | FileCheck %s

// Types that have cyclic references.

// CHECK: !testd.self_referencing<i32>
func.func @no_references(%v: !testd.self_referencing<i32>) {
  return
}

// -----

// CHECK: !testd.self_referencing<!testd.self_referencing<i32>>
func.func @one_reference(%v: !testd.self_referencing<!testd.self_referencing<i32>>) {
  return
}

// -----

// expected-error@+1 {{'i64' does not satisfy the constraint}}
func.func @wrong_parameter(%v: !testd.self_referencing<i64>) {
  return
}

// -----

// CHECK: !testd.type1<i32>
func.func @type1_no_references(%v: !testd.type1<i32>) {
  return
}

// -----

// CHECK: !testd.type1<!testd.type2<i32>>
func.func @type1_one_references(%v: !testd.type1<!testd.type2<i32>>) {
  return
}

// -----

// CHECK: !testd.type1<!testd.type2<!testd.type1<i32>>>
func.func @type1_two_references(%v: !testd.type1<!testd.type2<!testd.type1<i32>>>) {
  return
}

// -----

// expected-error@+1 {{'i64' does not satisfy the constraint}}
func.func @wrong_parameter_type1(%v: !testd.type1<i64>) {
  return
}

// -----

// expected-error@+1 {{'i64' does not satisfy the constraint}}
func.func @wrong_parameter_type2(%v: !testd.type2<i64>) {
  return
}
