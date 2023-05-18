// RUN: mlir-opt %s -split-input-file --test-verify-uselistorder -verify-diagnostics

// COM: --test-verify-uselistorder will randomly shuffle the uselist of every
//      value and do a roundtrip to bytecode. An error is returned if the
//      uselist order are not preserved when doing a roundtrip to bytecode. The
//      test needs to verify diagnostics to be functional.

func.func @base_test(%arg0 : i32) -> i32 {
  %0 = arith.constant 45 : i32
  %1 = arith.constant 46 : i32
  %2 = "test.addi"(%arg0, %arg0) : (i32, i32) -> i32
  %3 = "test.addi"(%2, %0) : (i32, i32) -> i32
  %4 = "test.addi"(%2, %1) : (i32, i32) -> i32
  %5 = "test.addi"(%3, %4) : (i32, i32) -> i32
  %6 = "test.addi"(%5, %4) : (i32, i32) -> i32
  %7 = "test.addi"(%6, %4) : (i32, i32) -> i32
  return %7 : i32
}

// -----

func.func @test_with_multiple_uses_in_same_op(%arg0 : i32) -> i32 {
  %0 = arith.constant 45 : i32
  %1 = arith.constant 46 : i32
  %2 = "test.addi"(%arg0, %arg0) : (i32, i32) -> i32
  %3 = "test.addi"(%2, %0) : (i32, i32) -> i32
  %4 = "test.addi"(%2, %1) : (i32, i32) -> i32
  %5 = "test.addi"(%2, %2) : (i32, i32) -> i32
  %6 = "test.addi"(%3, %4) : (i32, i32) -> i32
  %7 = "test.addi"(%6, %5) : (i32, i32) -> i32
  %8 = "test.addi"(%7, %4) : (i32, i32) -> i32
  %9 = "test.addi"(%8, %4) : (i32, i32) -> i32
  return %9 : i32
}

// -----

func.func @test_with_multiple_block_arg_uses(%arg0 : i32) -> i32 {
  %0 = arith.constant 45 : i32
  %1 = arith.constant 46 : i32
  %2 = "test.addi"(%arg0, %arg0) : (i32, i32) -> i32
  %3 = "test.addi"(%2, %arg0) : (i32, i32) -> i32
  %4 = "test.addi"(%2, %1) : (i32, i32) -> i32
  %5 = "test.addi"(%2, %2) : (i32, i32) -> i32
  %6 = "test.addi"(%3, %4) : (i32, i32) -> i32
  %7 = "test.addi"(%6, %5) : (i32, i32) -> i32
  %8 = "test.addi"(%7, %4) : (i32, i32) -> i32
  %9 = "test.addi"(%8, %4) : (i32, i32) -> i32
  return %9 : i32
}

// -----

// Test that use-lists in region with no dominance are preserved
test.graph_region {
  %0 = "test.foo"(%1) : (i32) -> i32
  test.graph_region attributes {a} {
    %a = "test.a"(%b) : (i32) -> i32
    %b = "test.b"(%2) : (i32) -> i32
  }
  %1 = "test.bar"(%2) : (i32) -> i32
  %2 = "test.baz"() : () -> i32
}
