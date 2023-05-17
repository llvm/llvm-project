// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize{top-down=true}))' | FileCheck %s --check-prefix=TD
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s --check-prefix=BU


// BU-LABEL: func @default_insertion_position
// TD-LABEL: func @default_insertion_position
func.func @default_insertion_position(%cond: i1) {
  // Constant should be folded into the entry block.

  // BU: arith.constant 2
  // BU-NEXT: scf.if

  // TD: arith.constant 2
  // TD-NEXT: scf.if
  scf.if %cond {
    %0 = arith.constant 1 : i32
    %2 = arith.addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }
  return
}

// This shows that we don't pull the constant out of the region because it
// wants to be the insertion point for the constant.
// BU-LABEL: func @custom_insertion_position
// TD-LABEL: func @custom_insertion_position
func.func @custom_insertion_position() {
  // BU: test.one_region_op
  // BU-NEXT: arith.constant 2

  // TD: test.one_region_op
  // TD-NEXT: arith.constant 2
  "test.one_region_op"() ({

    %0 = arith.constant 1 : i32
    %2 = arith.addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }) : () -> ()
  return
}

