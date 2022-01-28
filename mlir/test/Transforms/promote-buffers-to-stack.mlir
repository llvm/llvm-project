// RUN: mlir-opt -promote-buffers-to-stack -split-input-file %s | FileCheck %s --check-prefix=CHECK --check-prefix DEFINDEX
// RUN: mlir-opt -promote-buffers-to-stack="max-alloc-size-in-bytes=64" -split-input-file %s | FileCheck %s --check-prefix=CHECK --check-prefix LOWLIMIT
// RUN: mlir-opt -promote-buffers-to-stack="max-rank-of-allocated-memref=2" -split-input-file %s | FileCheck %s --check-prefix=CHECK --check-prefix RANK

// This file checks the behavior of PromoteBuffersToStack pass for converting
// AllocOps into AllocaOps, if possible.

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// PromoteBuffersToStack expected behavior: It should convert %0 into an
// AllocaOp.

// CHECK-LABEL: func @condBranch
func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cond_br {{.*}}
//      CHECK: ^bb2
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
//      CHECK: test.copy
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// PromoteBuffersToStack expected behavior:
// Since the alloc has dynamic type, it is not converted into an alloca.

// CHECK-LABEL: func @condBranchDynamicType
func @condBranchDynamicType(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  br ^bb3(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = memref.alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  br ^bb3(%1 : memref<?xf32>)
^bb3(%2: memref<?xf32>):
  test.copy(%2, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-NEXT: cond_br
//      CHECK: ^bb2
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc(%[[IDX]])
// CHECK-NEXT: test.buffer_based
//      CHECK: br ^bb3
// CHECK-NEXT: ^bb3(%[[ALLOC0:.*]]:{{.*}})
//      CHECK: test.copy(%[[ALLOC0]],
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @dynamicRanked
func @dynamicRanked(%memref: memref<*xf32>) {
  %0 = memref.rank %memref : memref<*xf32>
  %1 = memref.alloc(%0) : memref<?xindex>
  return
}

// CHECK-NEXT: %[[RANK:.*]] = memref.rank %{{.*}} : memref<*xf32>
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca(%[[RANK]])

// -----

// CHECK-LABEL: func @dynamicRanked2D
func @dynamicRanked2D(%memref: memref<*xf32>) {
  %0 = memref.rank %memref : memref<*xf32>
  %1 = memref.alloc(%0, %0) : memref<?x?xindex>
  return
}

// CHECK-NEXT: %[[RANK:.*]] = memref.rank %{{.*}} : memref<*xf32>
//  RANK-NEXT: %[[ALLOC:.*]] = memref.alloca(%[[RANK]], %[[RANK]])
// DEFINDEX-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[RANK]], %[[RANK]])

// -----

// CHECK-LABEL: func @dynamicNoRank
func @dynamicNoRank(%arg0: index) {
  %0 = memref.alloc(%arg0) : memref<?xindex>
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc

// -----

// Test Case: Existing AllocOp with no users.
// PromoteBuffersToStack expected behavior: It should convert it to an
// AllocaOp.

// CHECK-LABEL: func @emptyUsesValue
func @emptyUsesValue(%arg0: memref<4xf32>) {
  %0 = memref.alloc() : memref<4xf32>
  return
}
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0
//   /   \
//  |    bb1 <- Initial position of AllocOp
//   \   /
//    bb2
// PromoteBuffersToStack expected behavior: It should convert it into an
// AllocaOp.

// CHECK-LABEL: func @criticalEdge
func @criticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cond_br {{.*}}
//      CHECK: ^bb1
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
//      CHECK: test.copy
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0 <- Initial position of AllocOp
//   /   \
//  |    bb1
//   \   /
//    bb2
// PromoteBuffersToStack expected behavior: It converts the alloc in an alloca.

// CHECK-LABEL: func @invCriticalEdge
func @invCriticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
//      CHECK: cond_br
//      CHECK: test.copy
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0 <- Initial position of the first AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3 <- Initial position of the second AllocOp
// PromoteBuffersToStack expected behavior: It converts the allocs into allocas.

// CHECK-LABEL: func @ifElse
func @ifElse(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%5: memref<2xf32>) out(%7: memref<2xf32>)
  test.copy(%7, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA0:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOCA1:.*]] = memref.alloca()
//      CHECK: test.buffer_based
//      CHECK: test.copy(%[[ALLOCA1]]
// CHECK-NEXT: return

// -----

// Test Case: No users for buffer in if-else CFG
//    bb0 <- Initial position of AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3
// PromoteBuffersToStack expected behavior: It converts the alloc into alloca.

// CHECK-LABEL: func @ifElseNoUsers
func @ifElseNoUsers(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  test.copy(%arg1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
//      CHECK: return

// -----

// Test Case:
//      bb0 <- Initial position of the first AllocOp
//     /    \
//   bb1    bb2
//    |     /  \
//    |   bb3  bb4
//    \     \  /
//     \     /
//       bb5 <- Initial position of the second AllocOp
// PromoteBuffersToStack expected behavior: The two allocs should be converted
// into allocas.

// CHECK-LABEL: func @ifElseNested
func @ifElseNested(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb5(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cond_br %arg0, ^bb3(%3 : memref<2xf32>), ^bb4(%4 : memref<2xf32>)
^bb3(%5: memref<2xf32>):
  br ^bb5(%5, %3 : memref<2xf32>, memref<2xf32>)
^bb4(%6: memref<2xf32>):
  br ^bb5(%3, %6 : memref<2xf32>, memref<2xf32>)
^bb5(%7: memref<2xf32>, %8: memref<2xf32>):
  %9 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%9: memref<2xf32>)
  test.copy(%9, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA0:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOCA1:.*]] = memref.alloca()
//      CHECK: test.buffer_based
//      CHECK: test.copy(%[[ALLOCA1]]
// CHECK-NEXT: return

// -----

// Test Case: Dead operations in a single block.
// PromoteBuffersToStack expected behavior: It converts the two AllocOps into
// allocas.

// CHECK-LABEL: func @redundantOperations
func @redundantOperations(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

//      CHECK: (%[[ARG0:.*]]: {{.*}})
// CHECK-NEXT: %[[ALLOCA0:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}} out(%[[ALLOCA0]]
//      CHECK: %[[ALLOCA1:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based in(%[[ALLOCA0]]{{.*}} out(%[[ALLOCA1]]
//      CHECK: return

// -----

// Test Case:
//                                     bb0
//                                    /   \
// Initial pos of the 1st AllocOp -> bb1  bb2 <- Initial pos of the 2nd AllocOp
//                                    \   /
//                                     bb3
// PromoteBuffersToStack expected behavior: Both AllocOps are converted into
// allocas.

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func @moving_alloc_and_inserting_missing_dealloc(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  br ^exit(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  test.copy(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cond_br {{.*}}
//      CHECK: ^bb1
// CHECK-NEXT: %{{.*}} = memref.alloca()
//      CHECK: ^bb2
// CHECK-NEXT: %{{.*}} = memref.alloca()
//      CHECK: test.copy
// CHECK-NEXT: return

// -----

// Test Case: Nested regions - This test defines a BufferBasedOp inside the
// region of a RegionBufferBasedOp.
// PromoteBuffersToStack expected behavior: The AllocOps are converted into
// allocas.

// CHECK-LABEL: func @nested_regions_and_cond_branch
func @nested_regions_and_cond_branch(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = memref.alloc() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = math.exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT:   cond_br {{.*}}
//      CHECK:   ^bb2
// CHECK-NEXT:   %[[ALLOCA0:.*]] = memref.alloca()
//      CHECK:   ^bb0
// CHECK-NEXT:   %[[ALLOCA1:.*]] = memref.alloc()

// -----

// Test Case: buffer deallocation escaping
// PromoteBuffersToStack expected behavior: The first alloc is returned, so
// there is no conversion allowed. The second alloc is converted, since it
// only remains in the scope of the function.

// CHECK-LABEL: func @memref_in_function_results
func @memref_in_function_results(
  %arg0: memref<5xf32>,
  %arg1: memref<10xf32>,
  %arg2: memref<5xf32>) -> (memref<10xf32>, memref<15xf32>) {
  %x = memref.alloc() : memref<15xf32>
  %y = memref.alloc() : memref<5xf32>
  test.buffer_based in(%arg0: memref<5xf32>) out(%y: memref<5xf32>)
  test.copy(%y, %arg2) : (memref<5xf32>, memref<5xf32>)
  return %arg1, %x : memref<10xf32>, memref<15xf32>
}
//      CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>,
// CHECK-SAME: %[[RESULT:.*]]: memref<5xf32>)
//      CHECK: %[[ALLOC:.*]] = memref.alloc()
//      CHECK: %[[ALLOCA:.*]] = memref.alloca()
//      CHECK: test.copy
//      CHECK: return %[[ARG1]], %[[ALLOC]]

// -----

// Test Case: nested region control flow
// The allocation in the nested if branch cannot be converted to an alloca
// due to its dynamic memory allocation behavior.

// CHECK-LABEL: func @nested_region_control_flow
func @nested_region_control_flow(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.yield %1 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC0]]
//      CHECK: %[[ALLOC2:.*]] = memref.alloc(%arg0, %arg1)
// CHECK-NEXT: scf.yield %[[ALLOC0]]
//      CHECK: return %[[ALLOC1]]

// -----

// Test Case: nested region control flow within a region interface.
// The alloc %0 does not need to be converted in this case since the
// allocation finally escapes the method.

// CHECK-LABEL: func @inner_region_control_flow
func @inner_region_control_flow(%arg0 : index) -> memref<2x2xf32> {
  %0 = memref.alloc() : memref<2x2xf32>
  %1 = test.region_if %0 : memref<2x2xf32> -> (memref<2x2xf32>) then {
    ^bb0(%arg1 : memref<2x2xf32>):
      test.region_if_yield %arg1 : memref<2x2xf32>
  } else {
    ^bb0(%arg1 : memref<2x2xf32>):
      test.region_if_yield %arg1 : memref<2x2xf32>
  } join {
    ^bb0(%arg1 : memref<2x2xf32>):
      test.region_if_yield %arg1 : memref<2x2xf32>
  }
  return %1 : memref<2x2xf32>
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = test.region_if
// CHECK-NEXT: ^bb0(%[[ALLOC2:.*]]:{{.*}}):
// CHECK-NEXT: test.region_if_yield %[[ALLOC2]]
//      CHECK: ^bb0(%[[ALLOC3:.*]]:{{.*}}):
// CHECK-NEXT: test.region_if_yield %[[ALLOC3]]
//      CHECK: ^bb0(%[[ALLOC4:.*]]:{{.*}}):
// CHECK-NEXT: test.region_if_yield %[[ALLOC4]]
//      CHECK: return %[[ALLOC1]]

// -----

// Test Case: structured control-flow loop using a nested alloc.
// Alloc %0 will be converted to an alloca. %3 is not transformed.

// CHECK-LABEL: func @loop_alloc
func @loop_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = arith.cmpi eq, %i, %ub : index
    %3 = memref.alloc() : memref<2xf32>
    scf.yield %3 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
// CHECK-NEXT: scf.for
//      CHECK: %[[ALLOC:.*]] = memref.alloc()

// -----

// Test Case: structured control-flow loop with a nested if operation.
// The loop yields buffers that have been defined outside of the loop and the
// backedges only use the iteration arguments (or one of its aliases).
// Therefore, we do not have to (and are not allowed to) free any buffers
// that are passed via the backedges. The alloc is converted to an AllocaOp.

// CHECK-LABEL: func @loop_nested_if_no_alloc
func @loop_nested_if_no_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = arith.cmpi eq, %i, %ub : index
    %3 = scf.if %2 -> (memref<2xf32>) {
      scf.yield %0 : memref<2xf32>
    } else {
      scf.yield %iterBuf : memref<2xf32>
    }
    scf.yield %3 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOCA0:.*]] = memref.alloca()
// CHECK-NEXT: %[[ALLOCA1:.*]] = scf.for {{.*}} iter_args(%[[IALLOCA:.*]] =
//      CHECK: %[[ALLOCA2:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOCA0]]
//      CHECK: scf.yield %[[IALLOCA]]
//      CHECK: scf.yield %[[ALLOCA2]]
//      CHECK: test.copy(%[[ALLOCA1]], %arg4)

// -----

// Test Case: structured control-flow loop with a nested if operation using
// a deeply nested buffer allocation.
// The allocs are not converted in this case.

// CHECK-LABEL: func @loop_nested_if_alloc
func @loop_nested_if_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>) -> memref<2xf32> {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = arith.cmpi eq, %i, %ub : index
    %3 = scf.if %2 -> (memref<2xf32>) {
      %4 = memref.alloc() : memref<2xf32>
      scf.yield %4 : memref<2xf32>
    } else {
      scf.yield %0 : memref<2xf32>
    }
    scf.yield %3 : memref<2xf32>
  }
  return %1 : memref<2xf32>
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.for {{.*}}
//      CHECK: %[[ALLOC2:.*]] = scf.if
//      CHECK: %[[ALLOC3:.*]] = memref.alloc()
// CHECK-NEXT: scf.yield %[[ALLOC3]]
//      CHECK: scf.yield %[[ALLOC0]]
//      CHECK: scf.yield %[[ALLOC2]]
//      CHECK: return %[[ALLOC1]]

// -----

// Test Case: The allocated buffer is too large and, hence, it is not
// converted. In the actual implementation the largest size is 1KB.

// CHECK-LABEL: func @large_buffer_allocation
func @large_buffer_allocation(%arg0: memref<2048xf32>) {
  %0 = memref.alloc() : memref<2048xf32>
  test.copy(%0, %arg0) : (memref<2048xf32>, memref<2048xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.copy

// -----

// Test Case: AllocOp with element type index.
// PromoteBuffersToStack expected behavior: It should convert it to an
// AllocaOp.

// CHECK-LABEL: func @indexElementType
func @indexElementType() {
  %0 = memref.alloc() : memref<4xindex>
  return
}
// DEFINDEX-NEXT: memref.alloca()
// LOWLIMIT-NEXT: memref.alloca()
// RANK-NEXT: memref.alloca()
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @bigIndexElementType
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 256>>} {
  func @bigIndexElementType() {
    %0 = memref.alloc() : memref<4xindex>
    return
  }
}
// DEFINDEX-NEXT: memref.alloca()
// LOWLIMIT-NEXT: memref.alloc()
// RANK-NEXT: memref.alloca()
// CHECK-NEXT: return
