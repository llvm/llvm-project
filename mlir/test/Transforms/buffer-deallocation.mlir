// RUN: mlir-opt -buffer-deallocation -split-input-file %s | FileCheck %s

// This file checks the behaviour of BufferDeallocation pass for moving and
// inserting missing DeallocOps in their correct positions. Furthermore,
// copies and their corresponding AllocOps are inserted.

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferDeallocation expected behavior: bb2 contains an AllocOp which is
// passed to bb3. In the latter block, there should be an deallocation.
// Since bb1 does not contain an adequate alloc and the alloc in bb2 is not
// moved to bb0, we need to insert allocs and copies.

// CHECK-LABEL: func @condBranch
func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cond_br
//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: br ^bb3(%[[ALLOC0]]
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC1]]
// CHECK-NEXT: br ^bb3(%[[ALLOC2]]
//      CHECK: test.copy
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferDeallocation expected behavior: The existing AllocOp has a dynamic
// dependency to block argument %0 in bb2. Since the dynamic type is passed
// to bb3 via the block argument %2, it is currently required to allocate a
// temporary buffer for %2 that gets copies of %arg0 and %1 with their
// appropriate shape dimensions. The copy buffer deallocation will be applied
// to %2 in block bb3.

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
  %1 = alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  br ^bb3(%1 : memref<?xf32>)
^bb3(%2: memref<?xf32>):
  test.copy(%2, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-NEXT: cond_br
//      CHECK: %[[DIM0:.*]] = dim
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc(%[[DIM0]])
// CHECK-NEXT: linalg.copy(%{{.*}}, %[[ALLOC0]])
// CHECK-NEXT: br ^bb3(%[[ALLOC0]]
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc(%[[IDX]])
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[DIM1:.*]] = dim %[[ALLOC1]]
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc(%[[DIM1]])
// CHECK-NEXT: linalg.copy(%[[ALLOC1]], %[[ALLOC2]])
// CHECK-NEXT: dealloc %[[ALLOC1]]
// CHECK-NEXT: br ^bb3
// CHECK-NEXT: ^bb3(%[[ALLOC3:.*]]:{{.*}})
//      CHECK: test.copy(%[[ALLOC3]],
// CHECK-NEXT: dealloc %[[ALLOC3]]
// CHECK-NEXT: return

// -----

// Test Case:
//      bb0
//     /    \
//   bb1    bb2 <- Initial position of AllocOp
//    |     /  \
//    |   bb3  bb4
//    |     \  /
//    \     bb5
//     \    /
//       bb6
//        |
//       bb7
// BufferDeallocation expected behavior: The existing AllocOp has a dynamic
// dependency to block argument %0 in bb2. Since the dynamic type is passed to
// bb5 via the block argument %2 and to bb6 via block argument %3, it is
// currently required to allocate temporary buffers for %2 and %3 that gets
// copies of %1 and %arg0 1 with their appropriate shape dimensions. The copy
// buffer deallocations will be applied to %2 in block bb5 and to %3 in block
// bb6. Furthermore, there should be no copy inserted for %4.

// CHECK-LABEL: func @condBranchDynamicTypeNested
func @condBranchDynamicTypeNested(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  br ^bb6(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  cond_br %arg0, ^bb3, ^bb4
^bb3:
  br ^bb5(%1 : memref<?xf32>)
^bb4:
  br ^bb5(%1 : memref<?xf32>)
^bb5(%2: memref<?xf32>):
  br ^bb6(%2 : memref<?xf32>)
^bb6(%3: memref<?xf32>):
  br ^bb7(%3 : memref<?xf32>)
^bb7(%4: memref<?xf32>):
  test.copy(%4, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-NEXT: cond_br
//      CHECK: ^bb1
//      CHECK: %[[DIM0:.*]] = dim
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc(%[[DIM0]])
// CHECK-NEXT: linalg.copy(%{{.*}}, %[[ALLOC0]])
// CHECK-NEXT: br ^bb6
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc(%[[IDX]])
// CHECK-NEXT: test.buffer_based
//      CHECK: cond_br
//      CHECK: ^bb3:
// CHECK-NEXT: br ^bb5(%[[ALLOC1]]{{.*}})
//      CHECK: ^bb4:
// CHECK-NEXT: br ^bb5(%[[ALLOC1]]{{.*}})
// CHECK-NEXT: ^bb5(%[[ALLOC2:.*]]:{{.*}})
//      CHECK: %[[DIM2:.*]] = dim %[[ALLOC2]]
// CHECK-NEXT: %[[ALLOC3:.*]] = alloc(%[[DIM2]])
// CHECK-NEXT: linalg.copy(%[[ALLOC2]], %[[ALLOC3]])
// CHECK-NEXT: dealloc %[[ALLOC1]]
// CHECK-NEXT: br ^bb6(%[[ALLOC3]]{{.*}})
// CHECK-NEXT: ^bb6(%[[ALLOC4:.*]]:{{.*}})
// CHECK-NEXT: br ^bb7(%[[ALLOC4]]{{.*}})
// CHECK-NEXT: ^bb7(%[[ALLOC5:.*]]:{{.*}})
//      CHECK: test.copy(%[[ALLOC5]],
// CHECK-NEXT: dealloc %[[ALLOC4]]
// CHECK-NEXT: return

// -----

// Test Case: Existing AllocOp with no users.
// BufferDeallocation expected behavior: It should insert a DeallocOp right
// before ReturnOp.

// CHECK-LABEL: func @emptyUsesValue
func @emptyUsesValue(%arg0: memref<4xf32>) {
  %0 = alloc() : memref<4xf32>
  return
}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0
//   /   \
//  |    bb1 <- Initial position of AllocOp
//   \   /
//    bb2
// BufferDeallocation expected behavior: It should insert a DeallocOp at the
// exit block after CopyOp since %1 is an alias for %0 and %arg1. Furthermore,
// we have to insert a copy and an alloc in the beginning of the function.

// CHECK-LABEL: func @criticalEdge
func @criticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: cond_br
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC1]]
//      CHECK: test.copy
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0 <- Initial position of AllocOp
//   /   \
//  |    bb1
//   \   /
//    bb2
// BufferDeallocation expected behavior: It only inserts a DeallocOp at the
// exit block after CopyOp since %1 is an alias for %0 and %arg1.

// CHECK-LABEL: func @invCriticalEdge
func @invCriticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: dealloc
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0 <- Initial position of the first AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3 <- Initial position of the second AllocOp
// BufferDeallocation expected behavior: It only inserts two missing
// DeallocOps in the exit block. %5 is an alias for %0. Therefore, the
// DeallocOp for %0 should occur after the last BufferBasedOp. The Dealloc for
// %7 should happen after CopyOp.

// CHECK-LABEL: func @ifElse
func @ifElse(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = alloc() : memref<2xf32>
  test.buffer_based in(%5: memref<2xf32>) out(%7: memref<2xf32>)
  test.copy(%7, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: dealloc %[[FIRST_ALLOC]]
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: No users for buffer in if-else CFG
//    bb0 <- Initial position of AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3
// BufferDeallocation expected behavior: It only inserts a missing DeallocOp
// in the exit block since %5 or %6 are the latest aliases of %0.

// CHECK-LABEL: func @ifElseNoUsers
func @ifElseNoUsers(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
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

// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[FIRST_ALLOC]]
// CHECK-NEXT: return

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
// BufferDeallocation expected behavior: Two missing DeallocOps should be
// inserted in the exit block.

// CHECK-LABEL: func @ifElseNested
func @ifElseNested(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
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
  %9 = alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%9: memref<2xf32>)
  test.copy(%9, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: dealloc %[[FIRST_ALLOC]]
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: Dead operations in a single block.
// BufferDeallocation expected behavior: It only inserts the two missing
// DeallocOps after the last BufferBasedOp.

// CHECK-LABEL: func @redundantOperations
func @redundantOperations(%arg0: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  %1 = alloc() : memref<2xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

//      CHECK: (%[[ARG0:.*]]: {{.*}})
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[FIRST_ALLOC]]
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[FIRST_ALLOC]]{{.*}}out(%[[SECOND_ALLOC]]
//      CHECK: dealloc
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// Test Case:
//                                     bb0
//                                    /   \
// Initial pos of the 1st AllocOp -> bb1  bb2 <- Initial pos of the 2nd AllocOp
//                                    \   /
//                                     bb3
// BufferDeallocation expected behavior: We need to introduce a copy for each
// buffer since the buffers are passed to bb3. The both missing DeallocOps are
// inserted in the respective block of the allocs. The copy is freed in the exit
// block.

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func @moving_alloc_and_inserting_missing_dealloc(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  br ^exit(%0 : memref<2xf32>)
^bb2:
  %1 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  test.copy(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cond_br
//      CHECK: ^bb1
//      CHECK: ^bb1
//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC0]]
// CHECK-NEXT: br ^bb3(%[[ALLOC1]]
// CHECK-NEXT: ^bb2
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOC3:.*]] = alloc()
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC2]]
// CHECK-NEXT: br ^bb3(%[[ALLOC3]]
// CHECK-NEXT: ^bb3(%[[ALLOC4:.*]]:{{.*}})
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[ALLOC4]]
// CHECK-NEXT: return

// -----

// Test Case: Invalid position of the DeallocOp. There is a user after
// deallocation.
//   bb0
//  /   \
// bb1  bb2 <- Initial position of AllocOp
//  \   /
//   bb3
// BufferDeallocation expected behavior: The existing DeallocOp should be
// moved to exit block.

// CHECK-LABEL: func @moving_invalid_dealloc_op_complex
func @moving_invalid_dealloc_op_complex(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  %1 = alloc() : memref<2xf32>
  cond_br %cond, ^bb1, ^bb2
^bb1:
  br ^exit(%arg0 : memref<2xf32>)
^bb2:
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  dealloc %1 : memref<2xf32>
  br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  test.copy(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: cond_br
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[ALLOC0]]
// CHECK-NEXT: return

// -----

// Test Case: Inserting missing DeallocOp in a single block.

// CHECK-LABEL: func @inserting_missing_dealloc_simple
func @inserting_missing_dealloc_simple(
  %arg0 : memref<2xf32>,
  %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  test.copy(%0, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[ALLOC0]]

// -----

// Test Case: Moving invalid DeallocOp (there is a user after deallocation) in a
// single block.

// CHECK-LABEL: func @moving_invalid_dealloc_op
func @moving_invalid_dealloc_op(%arg0 : memref<2xf32>, %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  dealloc %0 : memref<2xf32>
  test.copy(%0, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[ALLOC0]]

// -----

// Test Case: Nested regions - This test defines a BufferBasedOp inside the
// region of a RegionBufferBasedOp.
// BufferDeallocation expected behavior: The AllocOp for the BufferBasedOp
// should remain inside the region of the RegionBufferBasedOp and it should insert
// the missing DeallocOp in the same region. The missing DeallocOp should be
// inserted after CopyOp.

// CHECK-LABEL: func @nested_regions_and_cond_branch
func @nested_regions_and_cond_branch(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = alloc() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}
//      CHECK: (%[[cond:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %{{.*}}: {{.*}})
// CHECK-NEXT:   cond_br %[[cond]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK:   %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT:   linalg.copy(%[[ARG1]], %[[ALLOC0]])
//      CHECK: ^[[BB2]]:
//      CHECK:   %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT:   test.region_buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
//      CHECK:     %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT:     test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC2]]
//      CHECK:     dealloc %[[ALLOC2]]
// CHECK-NEXT:     %{{.*}} = exp
//      CHECK:   %[[ALLOC3:.*]] = alloc()
// CHECK-NEXT:   linalg.copy(%[[ALLOC1]], %[[ALLOC3]])
// CHECK-NEXT:   dealloc %[[ALLOC1]]
//      CHECK:  ^[[BB3:.*]]({{.*}}):
//      CHECK:  test.copy
// CHECK-NEXT:  dealloc

// -----

// Test Case: buffer deallocation escaping
// BufferDeallocation expected behavior: It must not dealloc %arg1 and %x
// since they are operands of return operation and should escape from
// deallocating. It should dealloc %y after CopyOp.

// CHECK-LABEL: func @memref_in_function_results
func @memref_in_function_results(
  %arg0: memref<5xf32>,
  %arg1: memref<10xf32>,
  %arg2: memref<5xf32>) -> (memref<10xf32>, memref<15xf32>) {
  %x = alloc() : memref<15xf32>
  %y = alloc() : memref<5xf32>
  test.buffer_based in(%arg0: memref<5xf32>) out(%y: memref<5xf32>)
  test.copy(%y, %arg2) : (memref<5xf32>, memref<5xf32>)
  return %arg1, %x : memref<10xf32>, memref<15xf32>
}
//      CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>,
// CHECK-SAME: %[[RESULT:.*]]: memref<5xf32>)
//      CHECK: %[[X:.*]] = alloc()
//      CHECK: %[[Y:.*]] = alloc()
//      CHECK: test.copy
//      CHECK: dealloc %[[Y]]
//      CHECK: return %[[ARG1]], %[[X]]

// -----

// Test Case: nested region control flow
// The alloc %1 flows through both if branches until it is finally returned.
// Hence, it does not require a specific dealloc operation. However, %3
// requires a dealloc.

// CHECK-LABEL: func @nested_region_control_flow
func @nested_region_control_flow(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.yield %1 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC0]]
//      CHECK: %[[ALLOC2:.*]] = alloc(%arg0, %arg1)
// CHECK-NEXT: dealloc %[[ALLOC2]]
// CHECK-NEXT: scf.yield %[[ALLOC0]]
//      CHECK: return %[[ALLOC1]]

// -----

// Test Case: nested region control flow with a nested buffer allocation in a
// divergent branch.
// Buffer deallocation places a copy for both  %1 and %3, since they are
// returned in the end.

// CHECK-LABEL: func @nested_region_control_flow_div
func @nested_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.yield %3 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.if
//      CHECK: %[[ALLOC2:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC0]], %[[ALLOC2]])
//      CHECK: scf.yield %[[ALLOC2]]
//      CHECK: %[[ALLOC3:.*]] = alloc(%arg0, %arg1)
//      CHECK: %[[ALLOC4:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC3]], %[[ALLOC4]])
//      CHECK: dealloc %[[ALLOC3]]
//      CHECK: scf.yield %[[ALLOC4]]
//      CHECK: dealloc %[[ALLOC0]]
// CHECK-NEXT: return %[[ALLOC1]]

// -----

// Test Case: nested region control flow within a region interface.
// No copies are required in this case since the allocation finally escapes
// the method.

// CHECK-LABEL: func @inner_region_control_flow
func @inner_region_control_flow(%arg0 : index) -> memref<?x?xf32> {
  %0 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = test.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>) then {
    ^bb0(%arg1 : memref<?x?xf32>):
      test.region_if_yield %arg1 : memref<?x?xf32>
  } else {
    ^bb0(%arg1 : memref<?x?xf32>):
      test.region_if_yield %arg1 : memref<?x?xf32>
  } join {
    ^bb0(%arg1 : memref<?x?xf32>):
      test.region_if_yield %arg1 : memref<?x?xf32>
  }
  return %1 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = test.region_if
// CHECK-NEXT: ^bb0(%[[ALLOC2:.*]]:{{.*}}):
// CHECK-NEXT: test.region_if_yield %[[ALLOC2]]
//      CHECK: ^bb0(%[[ALLOC3:.*]]:{{.*}}):
// CHECK-NEXT: test.region_if_yield %[[ALLOC3]]
//      CHECK: ^bb0(%[[ALLOC4:.*]]:{{.*}}):
// CHECK-NEXT: test.region_if_yield %[[ALLOC4]]
//      CHECK: return %[[ALLOC1]]

// -----

// CHECK-LABEL: func @subview
func @subview(%arg0 : index, %arg1 : index, %arg2 : memref<?x?xf32>) {
  %0 = alloc() : memref<64x4xf32, offset: 0, strides: [4, 1]>
  %1 = subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
  to memref<?x?xf32, offset: ?, strides: [?, ?]>
  test.copy(%1, %arg2) :
    (memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: subview
// CHECK-NEXT: test.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: In the presence of AllocaOps only the AllocOps has top be freed.
// Therefore, all allocas are not handled.

// CHECK-LABEL: func @condBranchAlloca
func @condBranchAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloca() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cond_br
//      CHECK: %[[ALLOCA:.*]] = alloca()
//      CHECK: br ^bb3(%[[ALLOCA:.*]])
// CHECK-NEXT: ^bb3
// CHECK-NEXT: test.copy
// CHECK-NEXT: return

// -----

// Test Case: In the presence of AllocaOps only the AllocOps has top be freed.
// Therefore, all allocas are not handled. In this case, only alloc %0 has a
// dealloc.

// CHECK-LABEL: func @ifElseAlloca
func @ifElseAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = alloca() : memref<2xf32>
  test.buffer_based in(%5: memref<2xf32>) out(%7: memref<2xf32>)
  test.copy(%7, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOCA:.*]] = alloca()
// CHECK-NEXT: test.buffer_based
//      CHECK: dealloc %[[ALLOC]]
//      CHECK: test.copy
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @ifElseNestedAlloca
func @ifElseNestedAlloca(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  %0 = alloca() : memref<2xf32>
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
  %9 = alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%9: memref<2xf32>)
  test.copy(%9, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA:.*]] = alloca()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: test.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @nestedRegionsAndCondBranchAlloca
func @nestedRegionsAndCondBranchAlloca(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = alloca() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}
//      CHECK: (%[[cond:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %{{.*}}: {{.*}})
// CHECK-NEXT:   cond_br %[[cond]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: linalg.copy
//      CHECK: ^[[BB2]]:
//      CHECK:   %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT:   test.region_buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
//      CHECK:     %[[ALLOCA:.*]] = alloca()
// CHECK-NEXT:     test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOCA]]
//      CHECK:     %{{.*}} = exp
//      CHECK:  %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT:  linalg.copy
// CHECK-NEXT:  dealloc %[[ALLOC1]]
//      CHECK:  ^[[BB3:.*]]({{.*}}):
//      CHECK:  test.copy
// CHECK-NEXT:  dealloc

// -----

// CHECK-LABEL: func @nestedRegionControlFlowAlloca
func @nestedRegionControlFlowAlloca(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = alloca(%arg0, %arg1) : memref<?x?xf32>
    scf.yield %1 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC0]]
//      CHECK: %[[ALLOCA:.*]] = alloca(%arg0, %arg1)
// CHECK-NEXT: scf.yield %[[ALLOC0]]
//      CHECK: return %[[ALLOC1]]

// -----

// Test Case: structured control-flow loop using a nested alloc.
// The iteration argument %iterBuf has to be freed before yielding %3 to avoid
// memory leaks.

// CHECK-LABEL: func @loop_alloc
func @loop_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = cmpi "eq", %i, %ub : index
    %3 = alloc() : memref<2xf32>
    scf.yield %3 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: dealloc %[[ALLOC0]]
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
//      CHECK: linalg.copy(%arg3, %[[ALLOC1]])
//      CHECK: %[[ALLOC2:.*]] = scf.for {{.*}} iter_args
// CHECK-SAME: (%[[IALLOC:.*]] = %[[ALLOC1]]
//      CHECK:    cmpi
//      CHECK:    dealloc %[[IALLOC]]
//      CHECK:    %[[ALLOC3:.*]] = alloc()
//      CHECK:    %[[ALLOC4:.*]] = alloc()
//      CHECK:    linalg.copy(%[[ALLOC3]], %[[ALLOC4]])
//      CHECK:    dealloc %[[ALLOC3]]
//      CHECK:    scf.yield %[[ALLOC4]]
//      CHECK: }
//      CHECK: test.copy(%[[ALLOC2]], %arg4)
// CHECK-NEXT: dealloc %[[ALLOC2]]

// -----

// Test Case: structured control-flow loop with a nested if operation.
// The loop yields buffers that have been defined outside of the loop and the
// backeges only use the iteration arguments (or one of its aliases).
// Therefore, we do not have to (and are not allowed to) free any buffers
// that are passed via the backedges.

// CHECK-LABEL: func @loop_nested_if_no_alloc
func @loop_nested_if_no_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = cmpi "eq", %i, %ub : index
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

//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.for {{.*}} iter_args(%[[IALLOC:.*]] =
//      CHECK: %[[ALLOC2:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC0]]
//      CHECK: scf.yield %[[IALLOC]]
//      CHECK: scf.yield %[[ALLOC2]]
//      CHECK: test.copy(%[[ALLOC1]], %arg4)
//      CHECK: dealloc %[[ALLOC0]]

// -----

// Test Case: structured control-flow loop with a nested if operation using
// a deeply nested buffer allocation.
// Since the innermost allocation happens in a divergent branch, we have to
// introduce additional copies for the nested if operation. Since the loop's
// yield operation "returns" %3, it will return a newly allocated buffer.
// Therefore, we have to free the iteration argument %iterBuf before
// "returning" %3.

// CHECK-LABEL: func @loop_nested_if_alloc
func @loop_nested_if_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>) -> memref<2xf32> {
  %0 = alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = cmpi "eq", %i, %ub : index
    %3 = scf.if %2 -> (memref<2xf32>) {
      %4 = alloc() : memref<2xf32>
      scf.yield %4 : memref<2xf32>
    } else {
      scf.yield %0 : memref<2xf32>
    }
    scf.yield %3 : memref<2xf32>
  }
  return %1 : memref<2xf32>
}

//      CHECK: %[[ALLOC0:.*]] = alloc()
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%arg3, %[[ALLOC1]])
// CHECK-NEXT: %[[ALLOC2:.*]] = scf.for {{.*}} iter_args
// CHECK-SAME: (%[[IALLOC:.*]] = %[[ALLOC1]]
//      CHECK: dealloc %[[IALLOC]]
//      CHECK: %[[ALLOC3:.*]] = scf.if

//      CHECK: %[[ALLOC4:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC5:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC4]], %[[ALLOC5]])
// CHECK-NEXT: dealloc %[[ALLOC4]]
// CHECK-NEXT: scf.yield %[[ALLOC5]]

//      CHECK: %[[ALLOC6:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC0]], %[[ALLOC6]])
// CHECK-NEXT: scf.yield %[[ALLOC6]]

//      CHECK: %[[ALLOC7:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC3:.*]], %[[ALLOC7]])
// CHECK-NEXT: dealloc %[[ALLOC3]]
// CHECK-NEXT: scf.yield %[[ALLOC7]]

//      CHECK: dealloc %[[ALLOC0]]
// CHECK-NEXT: return %[[ALLOC2]]

// -----

// Test Case: several nested structured control-flow loops with a deeply nested
// buffer allocation inside an if operation.
// Same behavior is an loop_nested_if_alloc: we have to insert deallocations
// before each yield in all loops recursively.

// CHECK-LABEL: func @loop_nested_alloc
func @loop_nested_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = scf.for %i2 = %lb to %ub step %step
      iter_args(%iterBuf2 = %iterBuf) -> memref<2xf32> {
      %3 = scf.for %i3 = %lb to %ub step %step
        iter_args(%iterBuf3 = %iterBuf2) -> memref<2xf32> {
        %4 = alloc() : memref<2xf32>
        %5 = cmpi "eq", %i, %ub : index
        %6 = scf.if %5 -> (memref<2xf32>) {
          %7 = alloc() : memref<2xf32>
          scf.yield %7 : memref<2xf32>
        } else {
          scf.yield %iterBuf3 : memref<2xf32>
        }
        scf.yield %6 : memref<2xf32>
      }
      scf.yield %3 : memref<2xf32>
    }
    scf.yield %2 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: dealloc %[[ALLOC0]]
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%arg3, %[[ALLOC1]])
// CHECK-NEXT: %[[VAL_7:.*]] = scf.for {{.*}} iter_args
// CHECK-SAME: (%[[IALLOC0:.*]] = %[[ALLOC1]])
//      CHECK: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[IALLOC0]], %[[ALLOC2]])
// CHECK-NEXT: dealloc %[[IALLOC0]]
// CHECK-NEXT: %[[ALLOC3:.*]] = scf.for {{.*}} iter_args
// CHECK-SAME: (%[[IALLOC1:.*]] = %[[ALLOC2]])
//      CHECK: %[[ALLOC5:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[IALLOC1]], %[[ALLOC5]])
// CHECK-NEXT: dealloc %[[IALLOC1]]

//      CHECK: %[[ALLOC6:.*]] = scf.for {{.*}} iter_args
// CHECK-SAME: (%[[IALLOC2:.*]] = %[[ALLOC5]])
//      CHECK: %[[ALLOC8:.*]] = alloc()
// CHECK-NEXT: dealloc %[[ALLOC8]]
//      CHECK: %[[ALLOC9:.*]] = scf.if

//      CHECK: %[[ALLOC11:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC12:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC11]], %[[ALLOC12]])
// CHECK-NEXT: dealloc %[[ALLOC11]]
// CHECK-NEXT: scf.yield %[[ALLOC12]]

//      CHECK: %[[ALLOC13:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[IALLOC2]], %[[ALLOC13]])
// CHECK-NEXT: scf.yield %[[ALLOC13]]

//      CHECK: dealloc %[[IALLOC2]]
// CHECK-NEXT: %[[ALLOC10:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC9]], %[[ALLOC10]])
// CHECK-NEXT: dealloc %[[ALLOC9]]
// CHECK-NEXT: scf.yield %[[ALLOC10]]

//      CHECK: %[[ALLOC7:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC6]], %[[ALLOC7]])
// CHECK-NEXT: dealloc %[[ALLOC6]]
// CHECK-NEXT: scf.yield %[[ALLOC7]]

//      CHECK: %[[ALLOC4:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ALLOC3]], %[[ALLOC4]])
// CHECK-NEXT: dealloc %[[ALLOC3]]
// CHECK-NEXT: scf.yield %[[ALLOC4]]

//      CHECK: test.copy(%[[VAL_7]], %arg4)
// CHECK-NEXT: dealloc %[[VAL_7]]

// -----

// Test Case: explicit control-flow loop with a dynamically allocated buffer.
// The BufferDeallocation transformation should fail on this explicit
// control-flow loop since they are not supported.

// CHECK-LABEL: func @loop_dynalloc
func @loop_dynalloc(
  %arg0 : i32,
  %arg1 : i32,
  %arg2: memref<?xf32>,
  %arg3: memref<?xf32>) {
  %const0 = constant 0 : i32
  br ^loopHeader(%const0, %arg2 : i32, memref<?xf32>)

^loopHeader(%i : i32, %buff : memref<?xf32>):
  %lessThan = cmpi "slt", %i, %arg1 : i32
  cond_br %lessThan,
    ^loopBody(%i, %buff : i32, memref<?xf32>),
    ^exit(%buff : memref<?xf32>)

^loopBody(%val : i32, %buff2: memref<?xf32>):
  %const1 = constant 1 : i32
  %inc = addi %val, %const1 : i32
  %size = std.index_cast %inc : i32 to index
  %alloc1 = alloc(%size) : memref<?xf32>
  br ^loopHeader(%inc, %alloc1 : i32, memref<?xf32>)

^exit(%buff3 : memref<?xf32>):
  test.copy(%buff3, %arg3) : (memref<?xf32>, memref<?xf32>)
  return
}

// expected-error@+1 {{Structured control-flow loops are supported only}}

// -----

// Test Case: explicit control-flow loop with a dynamically allocated buffer.
// The BufferDeallocation transformation should fail on this explicit
// control-flow loop since they are not supported.

// CHECK-LABEL: func @do_loop_alloc
func @do_loop_alloc(
  %arg0 : i32,
  %arg1 : i32,
  %arg2: memref<2xf32>,
  %arg3: memref<2xf32>) {
  %const0 = constant 0 : i32
  br ^loopBody(%const0, %arg2 : i32, memref<2xf32>)

^loopBody(%val : i32, %buff2: memref<2xf32>):
  %const1 = constant 1 : i32
  %inc = addi %val, %const1 : i32
  %alloc1 = alloc() : memref<2xf32>
  br ^loopHeader(%inc, %alloc1 : i32, memref<2xf32>)

^loopHeader(%i : i32, %buff : memref<2xf32>):
  %lessThan = cmpi "slt", %i, %arg1 : i32
  cond_br %lessThan,
    ^loopBody(%i, %buff : i32, memref<2xf32>),
    ^exit(%buff : memref<2xf32>)

^exit(%buff3 : memref<2xf32>):
  test.copy(%buff3, %arg3) : (memref<2xf32>, memref<2xf32>)
  return
}

// expected-error@+1 {{Structured control-flow loops are supported only}}

// -----

// CHECK-LABEL: func @assumingOp(
func @assumingOp(
  %arg0: !shape.witness,
  %arg2: memref<2xf32>,
  %arg3: memref<2xf32>) {
  // Confirm the alloc will be dealloc'ed in the block.
  %1 = shape.assuming %arg0 -> memref<2xf32> {
     %0 = alloc() : memref<2xf32>
    shape.assuming_yield %arg2 : memref<2xf32>
  }
  // Confirm the alloc will be returned and dealloc'ed after its use.
  %3 = shape.assuming %arg0 -> memref<2xf32> {
    %2 = alloc() : memref<2xf32>
    shape.assuming_yield %2 : memref<2xf32>
  }
  test.copy(%3, %arg3) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: !shape.witness,
// CHECK-SAME: %[[ARG1:.*]]: {{.*}},
// CHECK-SAME: %[[ARG2:.*]]: {{.*}}
//      CHECK: %[[UNUSED_RESULT:.*]] = shape.assuming %[[ARG0]]
// CHECK-NEXT:    %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT:    dealloc %[[ALLOC0]]
// CHECK-NEXT:    shape.assuming_yield %[[ARG1]]
//      CHECK: %[[ASSUMING_RESULT:.*]] = shape.assuming %[[ARG0]]
// CHECK-NEXT:    %[[TMP_ALLOC:.*]] = alloc()
// CHECK-NEXT:    %[[RETURNING_ALLOC:.*]] = alloc()
// CHECK-NEXT:    linalg.copy(%[[TMP_ALLOC]], %[[RETURNING_ALLOC]])
// CHECK-NEXT:    dealloc %[[TMP_ALLOC]]
// CHECK-NEXT:    shape.assuming_yield %[[RETURNING_ALLOC]]
//      CHECK: test.copy(%[[ASSUMING_RESULT:.*]], %[[ARG2]])
// CHECK-NEXT: dealloc %[[ASSUMING_RESULT]]
