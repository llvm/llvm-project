// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Violations of the 1:1 relationship between the forwarded operands/results of
// a call operation and the arguments/results of its callee, as reported by
// `call_interface_impl::verifyCallOpInterface`.

// `test.call_and_produce` produces its first result itself; only the trailing
// results are forwarded from the callee. Neither the produced result nor a
// non-forwarded operand takes part in the verification.

func.func private @callee(i32) -> i32

func.func @too_many_forwarded_operands(%arg0: i32) {
  // expected-error @below {{incorrect number of operands for callee: expected 1, but got 2}}
  %status, %res = test.call_and_produce @callee(%arg0, %arg0) : (i32, i32) -> (i1, i32)
  return
}

// -----

func.func private @callee(i32, i32) -> i32

func.func @too_few_forwarded_operands(%arg0: i32) {
  // expected-error @below {{incorrect number of operands for callee: expected 2, but got 1}}
  %status, %res = test.call_and_produce @callee(%arg0) : (i32) -> (i1, i32)
  return
}

// -----

func.func private @callee(i32) -> i32

func.func @too_many_forwarded_results(%arg0: i32) {
  // expected-error @below {{incorrect number of results for callee: expected 1, but got 2}}
  %status, %res0, %res1 = test.call_and_produce @callee(%arg0) : (i32) -> (i1, i32, i32)
  return
}

// -----

func.func private @callee(i32) -> (i32, i32)

func.func @too_few_forwarded_results(%arg0: i32) {
  // expected-error @below {{incorrect number of results for callee: expected 2, but got 1}}
  %status, %res = test.call_and_produce @callee(%arg0) : (i32) -> (i1, i32)
  return
}

// -----

// The verifier requires corresponding types to be equal.

func.func private @callee(i32) -> i32

func.func @operand_types_must_match(%arg0: f32) {
  // expected-error @below {{operand type mismatch: expected operand type 'i32', but provided 'f32' for operand number 0}}
  %status, %res = test.call_and_produce @callee(%arg0) : (f32) -> (i1, i32)
  return
}

// -----

func.func private @callee(i32) -> i32

func.func @result_types_must_match(%arg0: i32) {
  // expected-error @+3 {{result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f64'}}
  // expected-note @+1 {{callee result types: 'i32'}}
  %status, %res = test.call_and_produce @callee(%arg0) : (i32) -> (i1, f64)
  return
}

// -----

// The declared parameters of a variadic callee are still checked. Here
// `var_callee_type` declares one parameter while the callee declares two, so
// only one operand is an argument operand and the call does not match.

llvm.func @printf(!llvm.ptr, i32, ...) -> i32

llvm.func @inconsistent_var_callee_type(%arg0: !llvm.ptr, %arg1: i32) {
  // expected-error @below {{incorrect number of operands for callee: expected 2, but got 1}}
  %res = llvm.call @printf(%arg0, %arg1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
  llvm.return
}

// -----

// A call operation that does not model variadic callees claims all its operands
// as argument operands, so it does not satisfy the 1:1 relationship here.

llvm.func @printf(!llvm.ptr, ...) -> i32

func.func @variadic_callee_not_modelled(%arg0: !llvm.ptr, %arg1: i32) {
  // expected-error @below {{incorrect number of operands for callee: expected 1, but got 2}}
  %status, %res = test.call_and_produce @printf(%arg0, %arg1) : (!llvm.ptr, i32) -> (i1, i32)
  return
}

// -----

// `func.call` can never be variadic: the builtin `FunctionType` has no variadic
// bit, so the number of forwarded operands always matches the callee exactly.

func.func private @callee(i32) -> i32

func.func @func_call_is_never_variadic(%arg0: i32) {
  // expected-error @below {{incorrect number of operands for callee}}
  %res = func.call @callee(%arg0, %arg0) : (i32, i32) -> i32
  return
}
