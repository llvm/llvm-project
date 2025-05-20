// RUN: mlir-opt --test-integer-lattice %s | FileCheck %s

// CHECK-LABEL: @test_returnlike
// CHECK: analysis_return_like_region_op
// CHECK-NEXT: arith.constant {test.operand_lattices = [], test.result_lattices = [0 : index]} 1 : i32
// CHECK-NEXT: region_yield
// CHECK-SAME: {test.operand_lattices = [0 : index], test.result_lattices = []}

// The core of the return-like test: the operand lattices of the yield forward
// to the result lattices of the enclosing region-holding op

// CHECK-NEXT: }) {test.operand_lattices = [], test.result_lattices = [0 : index]} : () -> i32
func.func @test_returnlike() {
  %0 = "test.analysis_return_like_region_op"() ({
    %0 = arith.constant 1 : i32
    "test.region_yield" (%0) : (i32) -> ()
  }) : () -> i32
  return
}
