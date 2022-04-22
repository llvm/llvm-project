// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @affine_apply_no_map() {
^bb0:
  %i = arith.constant 0 : index
  %x = "affine.apply" (%i) { } : (index) -> (index) //  expected-error {{requires attribute 'map'}}
  return
}

// -----

func.func @affine_apply_wrong_operand_count() {
^bb0:
  %i = arith.constant 0 : index
  %x = "affine.apply" (%i) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index) -> (index) //  expected-error {{'affine.apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

func.func @affine_apply_wrong_result_count() {
^bb0:
  %i = arith.constant 0 : index
  %j = arith.constant 1 : index
  %x = "affine.apply" (%i, %j) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index,index) -> (index) //  expected-error {{'affine.apply' op mapping must produce one value}}
  return
}

// -----

func.func @unknown_custom_op() {
^bb0:
  %i = test.crazyThing() {value = 0} : () -> index  // expected-error {{custom op 'test.crazyThing' is unknown}}
  return
}

// -----

func.func @unknown_std_op() {
  // expected-error@+1 {{unregistered operation 'func.foo_bar_op' found in dialect ('func') that does not allow unknown operations}}
  %0 = "func.foo_bar_op"() : () -> index
  return
}

// -----

func.func @calls(%arg0: i32) {
  %x = call @calls() : () -> i32  // expected-error {{incorrect number of operands for callee}}
  return
}

// -----

func.func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+2 {{different type than prior uses}}
  // expected-note@-2 {{prior use here}}
  %r = arith.select %cond, %t, %f : i32
}

// -----

func.func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+1 {{op operand #0 must be bool-like}}
  %r = "arith.select"(%cond, %t, %f) : (i32, i32, i32) -> i32
}

// -----

func.func @func_with_ops(i1, i32, i64) {
^bb0(%cond : i1, %t : i32, %f : i64):
  // TODO: expand post change in verification order. This is currently only
  // verifying that the type verification is failing but not the specific error
  // message. In final state the error should refer to mismatch in true_value and
  // false_value.
  // expected-error@+1 {{type}}
  %r = "arith.select"(%cond, %t, %f) : (i1, i32, i64) -> i32
}

// -----

func.func @func_with_ops(vector<12xi1>, vector<42xi32>, vector<42xi32>) {
^bb0(%cond : vector<12xi1>, %t : vector<42xi32>, %f : vector<42xi32>):
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "arith.select"(%cond, %t, %f) : (vector<12xi1>, vector<42xi32>, vector<42xi32>) -> vector<42xi32>
}

// -----

func.func @func_with_ops(tensor<12xi1>, tensor<42xi32>, tensor<42xi32>) {
^bb0(%cond : tensor<12xi1>, %t : tensor<42xi32>, %f : tensor<42xi32>):
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "arith.select"(%cond, %t, %f) : (tensor<12xi1>, tensor<42xi32>, tensor<42xi32>) -> tensor<42xi32>
}

// -----

func.func @return_not_in_function() {
  "foo.region"() ({
    // expected-error@+1 {{'func.return' op expects parent op 'func.func'}}
    return
  }): () -> ()
  return
}

// -----

func.func @invalid_splat(%v : f32) { // expected-note {{prior use here}}
  vector.splat %v : vector<8xf64>
  // expected-error@-1 {{expects different type than prior uses}}
  return
}

// -----

// Case that resulted in leak previously.

// expected-error@+1 {{expected ':' after block name}}
"g"()({^a:^b })
