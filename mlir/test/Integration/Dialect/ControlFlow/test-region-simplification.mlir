// Baseline check
// RUN: mlir-opt %s --convert-func-to-llvm --convert-cf-to-llvm | \
// RUN: mlir-cpu-runner -e nested_loop --entry-point-result=i32 | FileCheck %s

// Region simplification check
// RUN: mlir-opt %s \
// RUN: --canonicalize='enable-patterns=AnyPattern region-simplify=aggressive' \
// RUN: --convert-func-to-llvm --convert-cf-to-llvm | mlir-cpu-runner \
// RUN: -e nested_loop --entry-point-result=i32 | FileCheck %s

func.func @nested_loop() -> i32 {
  %c3_i64 = arith.constant 3 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  cf.br ^bb1(%c0_i32, %c0_i64 : i32, i64)
^bb1(%0: i32, %1: i64):  // 2 preds: ^bb0, ^bb8
  %2 = arith.cmpi ult, %1, %c2_i64 : i64
  cf.cond_br %2, ^bb2(%0, %1 : i32, i64), ^bb9(%0, %1 : i32, i64)
^bb2(%3: i32, %4: i64):  // pred: ^bb1
  %5 = arith.addi %4, %c1_i64 : i64
  cf.br ^bb3(%3, %5 : i32, i64)
^bb3(%6: i32, %7: i64):  // 2 preds: ^bb2, ^bb5
  %8 = arith.cmpi ult, %7, %c3_i64 : i64
  cf.cond_br %8, ^bb4(%6, %7 : i32, i64), ^bb6(%6, %7 : i32, i64)
^bb4(%9: i32, %10: i64):  // pred: ^bb3
  %11 = arith.addi %9, %c1_i32 : i32
  cf.br ^bb5(%11, %10 : i32, i64)
^bb5(%12: i32, %13: i64):  // pred: ^bb4
  %14 = arith.addi %13, %c1_i64 : i64
  cf.br ^bb3(%12, %14 : i32, i64)
^bb6(%15: i32, %16: i64):  // pred: ^bb3
  cf.br ^bb7
^bb7:  // pred: ^bb6
  cf.br ^bb8(%15, %4 : i32, i64)
^bb8(%17: i32, %18: i64):  // pred: ^bb7
  %19 = arith.addi %18, %c1_i64 : i64
  cf.br ^bb1(%17, %19 : i32, i64)
^bb9(%20: i32, %21: i64):  // pred: ^bb1
  cf.br ^bb10
^bb10:  // pred: ^bb9
  return %20 : i32
}

// If region simplification behaves correctly (by NOT merging ^bb2 and ^bb5),
// this will be 3.
// CHECK: 3
