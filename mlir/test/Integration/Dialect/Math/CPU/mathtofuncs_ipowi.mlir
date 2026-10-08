// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-runner --O3 -e test_zero_base_negative_exponent_i32 -entry-point-result=i32 | FileCheck %s --check-prefix=CHECK_TEST_ZERO_BASE_NEG_EXP_I32

// 0 ** -1 is 0. The expansion must not divide by zero to get there: on targets
// where integer division by zero traps, that is a SIGFPE instead of a result,
// and in any case it is UB that the optimizer is free to turn into the wrong
// value (hence --O3 on the RUN lines, so that this test exercises it).
func.func @test_zero_base_negative_exponent_i32() -> i32 {
  %base = arith.constant 0 : i32
  %exp = arith.constant -1 : i32
  %0 = math.ipowi %base, %exp : i32
  func.return %0 : i32
}
// CHECK_TEST_ZERO_BASE_NEG_EXP_I32: 0

// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-runner --O3 -e test_zero_base_negative_even_exponent_i64 -entry-point-result=i64 | FileCheck %s --check-prefix=CHECK_TEST_ZERO_BASE_NEG_EVEN_EXP_I64

// 0 ** -2 is 0 as well; the parity of the exponent must not matter.
func.func @test_zero_base_negative_even_exponent_i64() -> i64 {
  %base = arith.constant 0 : i64
  %exp = arith.constant -2 : i64
  %0 = math.ipowi %base, %exp : i64
  func.return %0 : i64
}
// CHECK_TEST_ZERO_BASE_NEG_EVEN_EXP_I64: 0

// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-runner --O3 -e test_minus_one_base_negative_odd_exponent_i32 -entry-point-result=i32 | FileCheck %s --check-prefix=CHECK_TEST_MINUS_ONE_BASE_NEG_ODD_EXP_I32

// The neighbouring negative-exponent cases keep working: (-1) ** -3 is -1.
func.func @test_minus_one_base_negative_odd_exponent_i32() -> i32 {
  %base = arith.constant -1 : i32
  %exp = arith.constant -3 : i32
  %0 = math.ipowi %base, %exp : i32
  func.return %0 : i32
}
// CHECK_TEST_MINUS_ONE_BASE_NEG_ODD_EXP_I32: -1

// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-runner --O3 -e test_positive_exponent_i32 -entry-point-result=i32 | FileCheck %s --check-prefix=CHECK_TEST_POSITIVE_EXP_I32

// Sanity check that the common path still computes: 2 ** 10 is 1024.
func.func @test_positive_exponent_i32() -> i32 {
  %base = arith.constant 2 : i32
  %exp = arith.constant 10 : i32
  %0 = math.ipowi %base, %exp : i32
  func.return %0 : i32
}
// CHECK_TEST_POSITIVE_EXP_I32: 1024
