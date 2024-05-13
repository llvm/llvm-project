// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e test_7i32_to_29 -entry-point-result=i32 | FileCheck %s --check-prefix=CHECK_TEST_7i32_TO_29

func.func @test_7i32_to_29() -> i32 {
  %arg = arith.constant 7 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK_TEST_7i32_TO_29: 29

// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e test_zero -entry-point-result=i32 | FileCheck %s --check-prefix=CHECK_TEST_ZERO

func.func @test_zero() -> i32 {
  %arg = arith.constant 0 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK_TEST_ZERO: 32

// Apparently mlir-cpu-runner doesn't support i8 return values, so testing i64 instead
// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e test_7i64_to_61 -entry-point-result=i64 | FileCheck %s --check-prefix=CHECK_TEST_7i64_TO_61

func.func @test_7i64_to_61() -> i64 {
  %arg = arith.constant 7 : i64
  %0 = math.ctlz %arg : i64
  func.return %0 : i64
}
// CHECK_TEST_7i64_TO_61: 61
