// RUN: mlir-opt %s \
// RUN:   --split-input-file \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-runner --split-input-file -entry-point-result=i32 \
// RUN: | FileCheck %s

func.func @main() -> i32 {
  %arg = arith.constant 7 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK: 29

// -----

func.func @main() -> i32 {
  %arg = arith.constant 0 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK: 32

// -----


// Apparently mlir-runner doesn't support i8 return values, so testing i64 instead
func.func @main() -> i32 {
  %arg = arith.constant 7 : i64
  %0 = math.ctlz %arg : i64
  // We can safely truncate 64-bit result to 32 bits
  %1 = arith.trunci %0 : i64 to i32
  func.return %1 : i32
}
// CHECK: 61
