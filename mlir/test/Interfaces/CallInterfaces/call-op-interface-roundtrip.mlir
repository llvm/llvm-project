// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Call operations whose forwarded operands/results are in a 1:1 relationship
// with the arguments/results of their callee, i.e., the cases that
// `call_interface_impl::verifyCallOpInterface` accepts.

// `test.call_and_produce` produces its first result itself; only the trailing
// results are forwarded from the callee. Neither the produced result nor a
// non-forwarded operand takes part in the verification.

func.func private @callee(i32) -> i32

// CHECK-LABEL: func @forwarded_operands_and_results_match(
//       CHECK:   test.call_and_produce @callee(%arg0) : (i32) -> (i1, i32)
func.func @forwarded_operands_and_results_match(%arg0: i32) {
  %status, %res = test.call_and_produce @callee(%arg0) : (i32) -> (i1, i32)
  return
}

// The produced result is not counted: a callee without results is fine even
// though the call operation has one result.

func.func private @callee_without_results(i32)

// CHECK-LABEL: func @produced_result_only(
//       CHECK:   test.call_and_produce @callee_without_results(%arg0) : (i32) -> i1
func.func @produced_result_only(%arg0: i32) {
  %status = test.call_and_produce @callee_without_results(%arg0) : (i32) -> i1
  return
}

// Nothing is verified if the callee cannot be resolved.

// CHECK-LABEL: func @unresolvable_callee(
//       CHECK:   test.call_and_produce @undefined_callee(%arg0) : (i32) -> (i1, f32)
func.func @unresolvable_callee(%arg0: i32) {
  %status, %res = test.call_and_produce @undefined_callee(%arg0) : (i32) -> (i1, f32)
  return
}

// Nothing is verified if the callee does not implement `CallableOpInterface`.

memref.global "private" @not_a_callable : memref<1xi32>

// CHECK-LABEL: func @callee_is_not_callable(
//       CHECK:   test.call_and_produce @not_a_callable(%arg0) : (i32) -> (i1, f32)
func.func @callee_is_not_callable(%arg0: i32) {
  %status, %res = test.call_and_produce @not_a_callable(%arg0) : (i32) -> (i1, f32)
  return
}

// The variadic arguments of a call to a variadic callee are consumed operands,
// not argument operands: `llvm.call` forwards only the declared parameter here,
// so the call satisfies the 1:1 relationship and passes the shared verifier.

llvm.func @printf(!llvm.ptr, ...) -> i32

// CHECK-LABEL: llvm.func @variadic_callee(
//       CHECK:   llvm.call @printf(%arg0, %arg1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
llvm.func @variadic_callee(%arg0: !llvm.ptr, %arg1: i32) {
  %res = llvm.call @printf(%arg0, %arg1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
  llvm.return
}
