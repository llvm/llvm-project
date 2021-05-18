// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func(canonicalize{top-down=true})' | FileCheck %s --check-prefix=TD
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func(canonicalize)' | FileCheck %s --check-prefix=BU


// BU-LABEL: func @default_insertion_position
// TD-LABEL: func @default_insertion_position
func @default_insertion_position(%cond: i1) {
  // Constant should be folded into the entry block.

  // BU: constant 2
  // BU-NEXT: scf.if

  // TD: constant 2
  // TD-NEXT: scf.if
  scf.if %cond {
    %0 = constant 1 : i32
    %2 = addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }
  return
}

// This shows that we don't pull the constant out of the region because it
// wants to be the insertion point for the constant.
// BU-LABEL: func @custom_insertion_position
// TD-LABEL: func @custom_insertion_position
func @custom_insertion_position() {
  // BU: test.one_region_op
  // BU-NEXT: constant 2

  // TD: test.one_region_op
  // TD-NEXT: constant 2
  "test.one_region_op"() ({

    %0 = constant 1 : i32
    %2 = addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }) : () -> ()
  return
}

