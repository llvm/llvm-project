// RUN: mlir-opt --canonicalize %s | FileCheck %s

// Test that control-flow cycles are not simplified infinitely.

// CHECK-LABEL: @cycle_2_blocks
// CHECK-NEXT: cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT: cf.br ^bb1
func.func @cycle_2_blocks() {
  cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb1
}

// CHECK-LABEL: @no_cycle_2_blocks
// CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT: return %c1_i32 : i32
func.func @no_cycle_2_blocks() -> i32 {
  cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb3
  ^bb3:
    %ret = arith.constant 1 : i32
    return %ret : i32
}

// CHECK-LABEL: @cycle_4_blocks
// CHECK-NEXT: cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT: cf.br ^bb1
func.func @cycle_4_blocks() {
  cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb3
  ^bb3:
    cf.br ^bb4
  ^bb4:
    cf.br ^bb1
}

// CHECK-LABEL: @no_cycle_4_blocks
// CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT: return %c1_i32 : i32
func.func @no_cycle_4_blocks() -> i32 {
  cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb3
  ^bb3:
    cf.br ^bb4
  ^bb4:
    cf.br ^bb5
  ^bb5:
    %ret = arith.constant 1 : i32
    return %ret : i32
}

// CHECK-LABEL: @delayed_3_cycle
// CHECK-NEXT: cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT: cf.br ^bb1
func.func @delayed_3_cycle() {
  cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb3
  ^bb3:
    cf.br ^bb4
  ^bb4:
    cf.br ^bb5
  ^bb5:
    cf.br ^bb3
}
