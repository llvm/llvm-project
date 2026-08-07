// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Tests `call_interface_impl::verifyCallOpInterface`, i.e., the 1:1
// relationship between the forwarded operands/results of a call operation and
// the arguments/results of its callee.

// `test.call_and_produce` produces its first result itself; only the trailing
// results are forwarded from the callee. Neither the produced result nor a
// non-forwarded operand takes part in the verification.

func.func private @callee(i32) -> i32

func.func @forwarded_operands_and_results_match(%arg0: i32) {
  %status, %res = test.call_and_produce @callee(%arg0) : (i32) -> (i1, i32)
  return
}

// -----

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

// The produced result is not counted: a callee without results is fine even
// though the call operation has one result.

func.func private @callee(i32)

func.func @produced_result_only(%arg0: i32) {
  %status = test.call_and_produce @callee(%arg0) : (i32) -> i1
  return
}

// -----

// Nothing is verified if the callee cannot be resolved.

func.func @unresolvable_callee(%arg0: i32) {
  %status, %res = test.call_and_produce @undefined_callee(%arg0) : (i32) -> (i1, f32)
  return
}

// -----

// Nothing is verified if the callee does not implement `CallableOpInterface`.

memref.global "private" @not_a_callable : memref<1xi32>

func.func @callee_is_not_callable(%arg0: i32) {
  %status, %res = test.call_and_produce @not_a_callable(%arg0) : (i32) -> (i1, f32)
  return
}

// -----

// By default, types must match across the call boundary.

func.func private @callee(i32) -> i32

func.func @default_operand_types_must_match(%arg0: f32) {
  // expected-error @below {{operand type mismatch: expected operand type 'i32', but provided 'f32' for operand number 0}}
  %status, %res = test.call_and_produce @callee(%arg0) : (f32) -> (i1, i32)
  return
}

// -----

func.func private @callee(i32) -> i32

func.func @default_result_types_must_match(%arg0: i32) {
  // expected-error @+3 {{result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f64'}}
  // expected-note @+1 {{callee result types: 'i32'}}
  %status, %res = test.call_and_produce @callee(%arg0) : (i32) -> (i1, f64)
  return
}

// -----

// `test.call_types_compat` implements `areTypesCompatible`: i32 and i64 are
// interchangeable, everything else must match.

func.func private @callee(i32) -> i32

func.func @compatible_types(%arg0: i64) {
  %res = test.call_types_compat @callee(%arg0) : (i64) -> i64
  return
}

// -----

func.func private @callee(i32) -> i32

func.func @incompatible_operand_type(%arg0: f32) {
  // expected-error @below {{operand type mismatch: expected operand type 'i32', but provided 'f32' for operand number 0}}
  %res = test.call_types_compat @callee(%arg0) : (f32) -> i32
  return
}

// -----

// The variadic arguments of a call to a variadic callee are consumed operands,
// not argument operands: `llvm.call` forwards only the declared parameter here,
// so the call satisfies the 1:1 relationship and passes the shared verifier.

llvm.func @printf(!llvm.ptr, ...) -> i32

llvm.func @variadic_callee(%arg0: !llvm.ptr, %arg1: i32) {
  %res = llvm.call @printf(%arg0, %arg1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
  llvm.return
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
  %res = test.call_types_compat @printf(%arg0, %arg1) : (!llvm.ptr, i32) -> i32
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

// -----

func.func private @callee(i32) -> i32

func.func @incompatible_result_type(%arg0: i32) {
  // expected-error @+3 {{result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f32'}}
  // expected-note @+1 {{callee result types: 'i32'}}
  %res = test.call_types_compat @callee(%arg0) : (i32) -> f32
  return
}
