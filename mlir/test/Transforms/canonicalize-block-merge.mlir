// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s

// Check the simple case of single operation blocks with a return.

// CHECK-LABEL: func @return_blocks(
func @return_blocks() {
  // CHECK: "foo.cond_br"()[^bb1, ^bb1]
  // CHECK: ^bb1:
  // CHECK-NEXT: return
  // CHECK-NOT: ^bb2

  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1:
  return
^bb2:
  return
}

// Check the case of identical blocks with matching arguments.

// CHECK-LABEL: func @matching_arguments(
func @matching_arguments() -> i32 {
  // CHECK: "foo.cond_br"()[^bb1, ^bb1]
  // CHECK: ^bb1(%{{.*}}: i32):
  // CHECK-NEXT: return
  // CHECK-NOT: ^bb2

  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1(%arg0 : i32):
  return %arg0 : i32
^bb2(%arg1 : i32):
  return %arg1 : i32
}

// Check that no merging occurs if there is an operand mismatch and we can't
// update th predecessor.

// CHECK-LABEL: func @mismatch_unknown_terminator
func @mismatch_unknown_terminator(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK: "foo.cond_br"()[^bb1, ^bb2]

  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1:
  return %arg0 : i32
^bb2:
  return %arg1 : i32
}

// Check that merging does occurs if there is an operand mismatch and we can
// update th predecessor.

// CHECK-LABEL: func @mismatch_operands
// CHECK-SAME: %[[COND:.*]]: i1, %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func @mismatch_operands(%cond : i1, %arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK: %[[RES:.*]] = select %[[COND]], %[[ARG0]], %[[ARG1]]
  // CHECK: return %[[RES]]

  cond_br %cond, ^bb1, ^bb2

^bb1:
  return %arg0 : i32
^bb2:
  return %arg1 : i32
}

// Check the same as above, but with pre-existing arguments.

// CHECK-LABEL: func @mismatch_operands_matching_arguments(
// CHECK-SAME: %[[COND:.*]]: i1, %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func @mismatch_operands_matching_arguments(%cond : i1, %arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK: %[[RES0:.*]] = select %[[COND]], %[[ARG1]], %[[ARG0]]
  // CHECK: %[[RES1:.*]] = select %[[COND]], %[[ARG0]], %[[ARG1]]
  // CHECK: return %[[RES1]], %[[RES0]]

  cond_br %cond, ^bb1(%arg1 : i32), ^bb2(%arg0 : i32)

^bb1(%arg2 : i32):
  return %arg0, %arg2 : i32, i32
^bb2(%arg3 : i32):
  return %arg1, %arg3 : i32, i32
}

// Check that merging does not occur if the uses of the arguments differ.

// CHECK-LABEL: func @mismatch_argument_uses(
func @mismatch_argument_uses(%cond : i1, %arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}}), ^bb2

  cond_br %cond, ^bb1(%arg1 : i32), ^bb2(%arg0 : i32)

^bb1(%arg2 : i32):
  return %arg0, %arg2 : i32, i32
^bb2(%arg3 : i32):
  return %arg3, %arg1 : i32, i32
}

// Check that merging does not occur if the types of the arguments differ.

// CHECK-LABEL: func @mismatch_argument_types(
func @mismatch_argument_types(%cond : i1, %arg0 : i32, %arg1 : i16) {
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}}), ^bb2

  cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg1 : i16)

^bb1(%arg2 : i32):
  "foo.return"(%arg2) : (i32) -> ()
^bb2(%arg3 : i16):
  "foo.return"(%arg3) : (i16) -> ()
}

// Check that merging does not occur if the number of the arguments differ.

// CHECK-LABEL: func @mismatch_argument_count(
func @mismatch_argument_count(%cond : i1, %arg0 : i32) {
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}}), ^bb2

  cond_br %cond, ^bb1(%arg0 : i32), ^bb2

^bb1(%arg2 : i32):
  "foo.return"(%arg2) : (i32) -> ()
^bb2:
  "foo.return"() : () -> ()
}

// Check that merging does not occur if the operations differ.

// CHECK-LABEL: func @mismatch_operations(
func @mismatch_operations(%cond : i1) {
  // CHECK: cond_br %{{.*}}, ^bb1, ^bb2

  cond_br %cond, ^bb1, ^bb2

^bb1:
  "foo.return"() : () -> ()
^bb2:
  return
}

// Check that merging does not occur if the number of operations differ.

// CHECK-LABEL: func @mismatch_operation_count(
func @mismatch_operation_count(%cond : i1) {
  // CHECK: cond_br %{{.*}}, ^bb1, ^bb2

  cond_br %cond, ^bb1, ^bb2

^bb1:
  "foo.op"() : () -> ()
  return
^bb2:
  return
}

// Check that merging does not occur if the blocks contain regions.

// CHECK-LABEL: func @contains_regions(
func @contains_regions(%cond : i1) {
  // CHECK: cond_br %{{.*}}, ^bb1, ^bb2

  cond_br %cond, ^bb1, ^bb2

^bb1:
  scf.if %cond {
    "foo.op"() : () -> ()
  }
  return
^bb2:
  scf.if %cond {
    "foo.op"() : () -> ()
  }
  return
}

// Check that properly handles back edges and the case where a value from one
// block is used in another.

// CHECK-LABEL: func @mismatch_loop(
// CHECK-SAME: %[[ARG:.*]]: i1, %[[ARG2:.*]]: i1
func @mismatch_loop(%cond : i1, %cond2 : i1) {
  // CHECK: cond_br %{{.*}}, ^bb1(%[[ARG2]] : i1), ^bb2

  cond_br %cond, ^bb2, ^bb3

^bb1:
  // CHECK: ^bb1(%[[ARG3:.*]]: i1):
  // CHECK-NEXT: %[[LOOP_CARRY:.*]] = "foo.op"
  // CHECK-NEXT: cond_br %[[ARG3]], ^bb1(%[[LOOP_CARRY]] : i1), ^bb2

  %ignored = "foo.op"() : () -> (i1)
  cond_br %cond3, ^bb1, ^bb3

^bb2:
  %cond3 = "foo.op"() : () -> (i1)
  cond_br %cond2, ^bb1, ^bb3

^bb3:
  // CHECK: ^bb2:
  // CHECK-NEXT: return

  return
}

// Check that blocks are not merged if the types of the operands differ.

// CHECK-LABEL: func @mismatch_operand_types(
func @mismatch_operand_types(%arg0 : i1, %arg1 : memref<i32>, %arg2 : memref<i1>) {
  %c0_i32 = constant 0 : i32
  %true = constant true
  br ^bb1

^bb1:
  cond_br %arg0, ^bb2, ^bb3

^bb2:
  // CHECK: store %{{.*}}, %{{.*}} : memref<i32>
  store %c0_i32, %arg1[] : memref<i32>
  br ^bb1

^bb3:
  // CHECK: store %{{.*}}, %{{.*}} : memref<i1>
  store %true, %arg2[] : memref<i1>
  br ^bb1
}
