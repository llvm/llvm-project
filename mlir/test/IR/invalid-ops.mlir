// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func @affine_apply_no_map() {
^bb0:
  %i = arith.constant 0 : index
  %x = "affine.apply" (%i) { } : (index) -> (index) //  expected-error {{requires attribute 'map'}}
  return
}

// -----

func @affine_apply_wrong_operand_count() {
^bb0:
  %i = arith.constant 0 : index
  %x = "affine.apply" (%i) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index) -> (index) //  expected-error {{'affine.apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

func @affine_apply_wrong_result_count() {
^bb0:
  %i = arith.constant 0 : index
  %j = arith.constant 1 : index
  %x = "affine.apply" (%i, %j) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index,index) -> (index) //  expected-error {{'affine.apply' op mapping must produce one value}}
  return
}

// -----

func @unknown_custom_op() {
^bb0:
  %i = test.crazyThing() {value = 0} : () -> index  // expected-error {{custom op 'test.crazyThing' is unknown}}
  return
}

// -----

func @unknown_std_op() {
  // expected-error@+1 {{unregistered operation 'std.foo_bar_op' found in dialect ('std') that does not allow unknown operations}}
  %0 = "std.foo_bar_op"() : () -> index
  return
}

// -----

func @calls(%arg0: i32) {
  %x = call @calls() : () -> i32  // expected-error {{incorrect number of operands for callee}}
  return
}

// -----

func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+2 {{different type than prior uses}}
  // expected-note@-2 {{prior use here}}
  %r = select %cond, %t, %f : i32
}

// -----

func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+1 {{op operand #0 must be bool-like}}
  %r = "std.select"(%cond, %t, %f) : (i32, i32, i32) -> i32
}

// -----

func @func_with_ops(i1, i32, i64) {
^bb0(%cond : i1, %t : i32, %f : i64):
  // TODO: expand post change in verification order. This is currently only
  // verifying that the type verification is failing but not the specific error
  // message. In final state the error should refer to mismatch in true_value and
  // false_value.
  // expected-error@+1 {{type}}
  %r = "std.select"(%cond, %t, %f) : (i1, i32, i64) -> i32
}

// -----

func @func_with_ops(vector<12xi1>, vector<42xi32>, vector<42xi32>) {
^bb0(%cond : vector<12xi1>, %t : vector<42xi32>, %f : vector<42xi32>):
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "std.select"(%cond, %t, %f) : (vector<12xi1>, vector<42xi32>, vector<42xi32>) -> vector<42xi32>
}

// -----

func @func_with_ops(tensor<12xi1>, tensor<42xi32>, tensor<42xi32>) {
^bb0(%cond : tensor<12xi1>, %t : tensor<42xi32>, %f : tensor<42xi32>):
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "std.select"(%cond, %t, %f) : (tensor<12xi1>, tensor<42xi32>, tensor<42xi32>) -> tensor<42xi32>
}

// -----

func @return_not_in_function() {
  "foo.region"() ({
    // expected-error@+1 {{'std.return' op expects parent op 'builtin.func'}}
    return
  }): () -> ()
  return
}

// -----

func @invalid_splat(%v : f32) {
  splat %v : memref<8xf32>
  // expected-error@-1 {{must be vector of any type values or statically shaped tensor of any type values}}
  return
}

// -----

func @invalid_splat(%v : vector<8xf32>) {
  %w = splat %v : tensor<8xvector<8xf32>>
  // expected-error@-1 {{must be integer/index/float type}}
  return
}

// -----

func @invalid_splat(%v : f32) { // expected-note {{prior use here}}
  splat %v : vector<8xf64>
  // expected-error@-1 {{expects different type than prior uses}}
  return
}

// -----

func @atomic_rmw_idxs_rank_mismatch(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  %x = atomic_rmw addf %val, %I[%i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func @atomic_rmw_expects_float(%I: memref<16x10xi32>, %i : index, %val : i32) {
  // expected-error@+1 {{expects a floating-point type}}
  %x = atomic_rmw addf %val, %I[%i, %i] : (i32, memref<16x10xi32>) -> i32
  return
}

// -----

func @atomic_rmw_expects_int(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects an integer type}}
  %x = atomic_rmw addi %val, %I[%i, %i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func @generic_atomic_rmw_wrong_arg_num(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected single number of entry block arguments}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%arg0 : f32, %arg1 : f32):
      %c1 = arith.constant 1.0 : f32
      atomic_yield %c1 : f32
  }
  return
}

// -----

func @generic_atomic_rmw_wrong_arg_type(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected block argument of the same type result type}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : i32):
      %c1 = arith.constant 1.0 : f32
      atomic_yield %c1 : f32
  }
  return
}

// -----

func @generic_atomic_rmw_result_type_mismatch(%I: memref<10xf32>, %i : index) {
 // expected-error@+1 {{failed to verify that result type matches element type of memref}}
 %0 = "std.generic_atomic_rmw"(%I, %i) ( {
    ^bb0(%old_value: f32):
      %c1 = arith.constant 1.0 : f32
      atomic_yield %c1 : f32
    }) : (memref<10xf32>, index) -> i32
  return
}

// -----

func @generic_atomic_rmw_has_side_effects(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{should contain only operations with no side effects}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1.0 : f32
      %buf = memref.alloc() : memref<2048xf32>
      atomic_yield %c1 : f32
  }
}

// -----

func @atomic_yield_type_mismatch(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{op types mismatch between yield op: 'i32' and its parent: 'f32'}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1 : i32
      atomic_yield %c1 : i32
  }
  return
}

// -----

func @no_zero_bit_integer_attrs() {
  // expected-error @+1 {{integer constant out of range for attribute}}
  %x = "some.op"(){value = 0 : i0} : () -> f32
  return
}
