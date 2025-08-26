// RUN: mlir-opt -allow-unregistered-dialect %s -test-inline-callback | FileCheck %s

// Test inlining with multiple blocks and scf.execute_region transformation
// CHECK-LABEL: func @test_inline_multiple_blocks
func.func @test_inline_multiple_blocks(%arg0: i32) -> i32 {
  // CHECK: %[[RES:.*]] = scf.execute_region -> i32
  // CHECK-NEXT: %[[ADD1:.*]] = arith.addi %arg0, %arg0
  // CHECK-NEXT: cf.br ^bb1(%[[ADD1]] : i32)
  // CHECK: ^bb1(%[[ARG:.*]]: i32):
  // CHECK-NEXT: %[[ADD2:.*]] = arith.addi %[[ARG]], %[[ARG]]
  // CHECK-NEXT: scf.yield %[[ADD2]]
  // CHECK: return %[[RES]]
  %fn = "test.functional_region_op"() ({
  ^bb0(%a : i32):
    %b = arith.addi %a, %a : i32
    cf.br ^bb1(%b: i32)
  ^bb1(%c: i32):
    %d = arith.addi %c, %c : i32
    "test.return"(%d) : (i32) -> ()
  }) : () -> ((i32) -> i32)

  %0 = call_indirect %fn(%arg0) : (i32) -> i32
  return %0 : i32
}
