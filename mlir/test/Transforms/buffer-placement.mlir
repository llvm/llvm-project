// RUN: mlir-opt -buffer-placement -split-input-file %s | FileCheck %s

// This file checks the behaviour of BufferPlacement pass for moving Alloc and
// Dealloc operations and inserting the missing the DeallocOps in their correct
// positions.

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferPlacement Expected Behaviour: It should move the existing AllocOp to
// the entry block, and insert a DeallocOp at the exit block after CopyOp since
// %1 is an alias for %0 and %arg1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @condBranch
func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: cond_br
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferPlacement Expected Behaviour: It should not move the existing AllocOp
// to any other block since the alloc has a dynamic dependency to block argument
// %0 in bb2. Since the dynamic type is passed to bb3 via the block argument %2,
// it is currently required to allocate a temporary buffer for %2 that gets
// copies of %arg0 and %1 with their appropriate shape dimensions. The copy
// buffer deallocation will be applied to %2 in block bb3.

#map0 = affine_map<(d0) -> (d0)>

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
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %1 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<?xf32>, memref<?xf32>
  br ^bb3(%1 : memref<?xf32>)
^bb3(%2: memref<?xf32>):
  "linalg.copy"(%2, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK-NEXT: cond_br
//      CHECK: %[[DIM0:.*]] = dim
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc(%[[DIM0]])
// CHECK-NEXT: linalg.copy(%{{.*}}, %[[ALLOC0]])
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc(%[[IDX]])
// CHECK-NEXT: linalg.generic
//      CHECK: %[[DIM1:.*]] = dim %[[ALLOC1]]
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc(%[[DIM1]])
// CHECK-NEXT: linalg.copy(%[[ALLOC1]], %[[ALLOC2]])
// CHECK-NEXT: dealloc %[[ALLOC1]]
// CHECK-NEXT: br ^bb3
// CHECK-NEXT: ^bb3(%[[ALLOC3:.*]]:{{.*}})
//      CHECK: linalg.copy(%[[ALLOC3]],
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
// BufferPlacement Expected Behaviour: It should not move the existing AllocOp
// to any other block since the alloc has a dynamic dependency to block argument
// %0 in bb2. Since the dynamic type is passed to bb5 via the block argument %2
// and to bb6 via block argument %3, it is currently required to allocate
// temporary buffers for %2 and %3 that gets copies of %1 and %arg0 1 with their
// appropriate shape dimensions. The copy buffer deallocations will be applied
// to %2 in block bb5 and to %3 in block bb6. Furthermore, there should be no
// copy inserted for %4.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @condBranchDynamicType
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
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %1 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<?xf32>, memref<?xf32>
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
  "linalg.copy"(%4, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK-NEXT: cond_br
//      CHECK: ^bb1
//      CHECK: %[[DIM0:.*]] = dim
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc(%[[DIM0]])
// CHECK-NEXT: linalg.copy(%{{.*}}, %[[ALLOC0]])
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc(%[[IDX]])
// CHECK-NEXT: linalg.generic
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
//      CHECK: linalg.copy(%[[ALLOC5]],
// CHECK-NEXT: dealloc %[[ALLOC4]]
// CHECK-NEXT: return

// -----

// Test Case: Existing AllocOp with no users.
// BufferPlacement Expected Behaviour: It should insert a DeallocOp right before
// ReturnOp.

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
// BufferPlacement Expected Behaviour: It should move the existing AllocOp to
// the entry block and insert a DeallocOp at the exit block after CopyOp since
// %1 is an alias for %0 and %arg1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @criticalEdge
func @criticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: cond_br
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case:
//    bb0 <- Initial position of AllocOp
//   /   \
//  |    bb1
//   \   /
//    bb2
// BufferPlacement Expected Behaviour: It shouldn't move the alloc position. It
// only inserts a DeallocOp at the exit block after CopyOp since %1 is an alias
// for %0 and %arg1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @invCriticalEdge
func @invCriticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
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
// BufferPlacement Expected Behaviour: It shouldn't move the AllocOps. It only
// inserts two missing DeallocOps in the exit block. %5 is an alias for %0.
// Therefore, the DeallocOp for %0 should occur after the last GenericOp. The
// Dealloc for %7 should happen after the CopyOp.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElse
func @ifElse(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %5, %7 {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  "linalg.copy"(%7, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic
//      CHECK: dealloc %[[FIRST_ALLOC]]
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: No users for buffer in if-else CFG
//    bb0 <- Initial position of AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3
// BufferPlacement Expected Behaviour: It shouldn't move the AllocOp. It only
// inserts a missing DeallocOp in the exit block since %5 or %6 are the latest
// aliases of %0.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElseNoUsers
func @ifElseNoUsers(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  "linalg.copy"(%arg1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
//      CHECK: dealloc %[[FIRST_ALLOC]]
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
// BufferPlacement Expected Behaviour: AllocOps shouldn't be moved.
// Two missing DeallocOps should be inserted in the exit block.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElseNested
func @ifElseNested(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
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
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %7, %9 {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  "linalg.copy"(%9, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic
//      CHECK: dealloc %[[FIRST_ALLOC]]
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: Dead operations in a single block.
// BufferPlacement Expected Behaviour: It shouldn't move the AllocOps. It only
// inserts the two missing DeallocOps after the last GenericOp.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @redundantOperations
func @redundantOperations(%arg0: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  %1 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %0, %1 {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  return
}

//      CHECK: (%[[ARG0:.*]]: {{.*}})
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ARG0]], %[[FIRST_ALLOC]]
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[FIRST_ALLOC]], %[[SECOND_ALLOC]]
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
// BufferPlacement Expected Behaviour: Both AllocOps should be moved to the
// entry block. Both missing DeallocOps should be moved to the exit block after
// CopyOp since %arg2 is an alias for %0 and %1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func @moving_alloc_and_inserting_missing_dealloc(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^exit(%0 : memref<2xf32>)
^bb2:
  %1 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %1 {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  "linalg.copy"(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %{{.*}} = alloc()
// CHECK-NEXT: %{{.*}} = alloc()
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// Test Case: Invalid position of the DeallocOp. There is a user after
// deallocation.
//   bb0
//  /   \
// bb1  bb2 <- Initial position of AllocOp
//  \   /
//   bb3
// BufferPlacement Expected Behaviour: It should move the AllocOp to the entry
// block. The existing DeallocOp should be moved to exit block.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @moving_invalid_dealloc_op_complex
func @moving_invalid_dealloc_op_complex(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  br ^exit(%arg0 : memref<2xf32>)
^bb2:
  %1 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %1 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  dealloc %1 : memref<2xf32>
  br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  "linalg.copy"(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %{{.*}} = alloc()
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// Test Case: Iserting missing DeallocOp in a single block.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @inserting_missing_dealloc_simple
func @inserting_missing_dealloc_simple(
  %arg0 : memref<2xf32>,
  %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  "linalg.copy"(%0, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

//      CHECK: linalg.copy
// CHECK-NEXT: dealloc

// -----

// Test Case: Moving invalid DeallocOp (there is a user after deallocation) in a
// single block.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @moving_invalid_dealloc_op
func @moving_invalid_dealloc_op(%arg0 : memref<2xf32>, %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  dealloc %0 : memref<2xf32>
  "linalg.copy"(%0, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

//      CHECK: linalg.copy
// CHECK-NEXT: dealloc

// -----

// Test Case: Nested regions - This test defines a GenericOp inside the region of
// another GenericOp.
// BufferPlacement Expected Behaviour: The AllocOp of inner GenericOp should remain
// inside the region of outer GenericOp and it should insert the missing DeallocOp
// in the same region. The AllocOp of the outer GenericOp should be moved to the
// entry block and its missing DeallocOp should be inserted after Linalg.Copy.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @nested_regions_and_cond_branch
func @nested_regions_and_cond_branch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = alloc() : memref<2xf32>
    linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %1 {
    ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
      %tmp2 = exp %gen2_arg0 : f32
      linalg.yield %tmp2 : f32
    }: memref<2xf32>, memref<2xf32>
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
//      CHECK: (%[[cond:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %{{.*}}: {{.*}})
// CHECK-NEXT:   %[[GENERIC1_ALLOC:.*]] = alloc()
// CHECK-NEXT:   cond_br %[[cond]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT:   linalg.generic {{{.*}}} %[[ARG1]], %[[GENERIC1_ALLOC]]
//      CHECK:     %[[GENERIC2_ALLOC:.*]] = alloc()
// CHECK-NEXT:     linalg.generic {{{.*}}} %[[ARG1]], %[[GENERIC2_ALLOC]]
//      CHECK:     dealloc %[[GENERIC2_ALLOC]]
// CHECK-NEXT:     %{{.*}} = exp
//      CHECK:  ^[[BB3:.*]]({{.*}}):
//      CHECK:  linalg.copy
// CHECK-NEXT:  dealloc %[[GENERIC1_ALLOC]]

// -----

// Test Case: buffer deallocation escaping
// BufferPlacement Expected Behaviour: It must not dealloc %arg1 and %x
// since they are operands of return operation and should escape from
// deallocating. It should dealloc %y after linalg.copy.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @memref_in_function_results
func @memref_in_function_results(%arg0: memref<5xf32>, %arg1: memref<10xf32>, %arg2: memref<5xf32>) -> (memref<10xf32>, memref<15xf32>) {
  %x = alloc() : memref<15xf32>
  %y = alloc() : memref<5xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %y {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = exp %arg3 : f32
    linalg.yield %2 : f32
  }: memref<5xf32>, memref<5xf32>
  linalg.copy(%y, %arg2) : memref<5xf32>, memref<5xf32>
  return %arg1, %x : memref<10xf32>, memref<15xf32>
}
// CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>, %[[RESULT:.*]]: memref<5xf32>)
// CHECK: %[[X:.*]] = alloc()
// CHECK: %[[Y:.*]] = alloc()
// CHECK: linalg.copy
// CHECK: dealloc %[[Y]]
// CHECK: return %[[ARG1]], %[[X]]

// -----

// Test Case: nested region control flow
// The alloc position of %1 does not need to be changed and flows through
// both if branches until it is finally returned. Hence, it does not
// require a specific dealloc operation. However, %3 requires a dealloc.

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
// The alloc positions of %1, %3 does not need to be changed since
// BufferPlacement does not move allocs out of nested regions at the moment.
// However, since %3 is allocated and "returned" in a divergent branch, we have
// to allocate a temporary buffer (like in condBranchDynamicTypeNested).

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

// Test Case: deeply nested region control flow with a nested buffer allocation
// in a divergent branch.
// The alloc positions of %1, %4 and %5 does not need to be changed since
// BufferPlacement does not move allocs out of nested regions at the moment.
// However, since %4 is allocated and "returned" in a divergent branch, we have
// to allocate several temporary buffers (like in condBranchDynamicTypeNested).

// CHECK-LABEL: func @nested_region_control_flow_div_nested
func @nested_region_control_flow_div_nested(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    %3 = scf.if %0 -> (memref<?x?xf32>) {
      scf.yield %1 : memref<?x?xf32>
    } else {
      %4 = alloc(%arg0, %arg1) : memref<?x?xf32>
      scf.yield %4 : memref<?x?xf32>
    }
    scf.yield %3 : memref<?x?xf32>
  } else {
    %5 = alloc(%arg1, %arg1) : memref<?x?xf32>
    scf.yield %5 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}
//      CHECK: %[[ALLOC0:.*]] = alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC2:.*]] = scf.if
//      CHECK: %[[ALLOC3:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC0]], %[[ALLOC3]])
//      CHECK: scf.yield %[[ALLOC3]]
//      CHECK: %[[ALLOC4:.*]] = alloc(%arg0, %arg1)
//      CHECK: %[[ALLOC5:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC4]], %[[ALLOC5]])
//      CHECK: dealloc %[[ALLOC4]]
//      CHECK: scf.yield %[[ALLOC5]]
//      CHECK: %[[ALLOC6:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC2]], %[[ALLOC6]])
//      CHECK: dealloc %[[ALLOC2]]
//      CHECK: scf.yield %[[ALLOC6]]
//      CHECK: %[[ALLOC7:.*]] = alloc(%arg1, %arg1)
//      CHECK: %[[ALLOC8:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC7]], %[[ALLOC8]])
//      CHECK: dealloc %[[ALLOC7]]
//      CHECK: scf.yield %[[ALLOC8]]
//      CHECK: dealloc %[[ALLOC0]]
// CHECK-NEXT: return %[[ALLOC1]]

// -----

// Test Case: nested region control flow within a region interface.
// The alloc positions of %0 does not need to be changed and no copies are
// required in this case since the allocation finally escapes the method.

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

// Test Case: nested region control flow within a region interface including an
// allocation in a divergent branch.
// The alloc positions of %1 and %2 does not need to be changed since
// BufferPlacement does not move allocs out of nested regions at the moment.
// However, since %2 is allocated and yielded in a divergent branch, we have
// to allocate several temporary buffers (like in condBranchDynamicTypeNested).

// CHECK-LABEL: func @inner_region_control_flow_div
func @inner_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = test.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>) then {
    ^bb0(%arg2 : memref<?x?xf32>):
      test.region_if_yield %arg2 : memref<?x?xf32>
  } else {
    ^bb0(%arg2 : memref<?x?xf32>):
      %2 = alloc(%arg0, %arg1) : memref<?x?xf32>
      test.region_if_yield %2 : memref<?x?xf32>
  } join {
    ^bb0(%arg2 : memref<?x?xf32>):
      test.region_if_yield %arg2 : memref<?x?xf32>
  }
  return %1 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = test.region_if
// CHECK-NEXT: ^bb0(%[[ALLOC2:.*]]:{{.*}}):
//      CHECK: %[[ALLOC3:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC2]], %[[ALLOC3]])
// CHECK-NEXT: test.region_if_yield %[[ALLOC3]]
//      CHECK: ^bb0(%[[ALLOC4:.*]]:{{.*}}):
//      CHECK: %[[ALLOC5:.*]] = alloc
//      CHECK: %[[ALLOC6:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC5]], %[[ALLOC6]])
// CHECK-NEXT: dealloc %[[ALLOC5]]
// CHECK-NEXT: test.region_if_yield %[[ALLOC6]]
//      CHECK: ^bb0(%[[ALLOC7:.*]]:{{.*}}):
//      CHECK: %[[ALLOC8:.*]] = alloc
// CHECK-NEXT: linalg.copy(%[[ALLOC7]], %[[ALLOC8]])
// CHECK-NEXT: dealloc %[[ALLOC7]]
// CHECK-NEXT: test.region_if_yield %[[ALLOC8]]
//      CHECK: dealloc %[[ALLOC0]]
// CHECK-NEXT: return %[[ALLOC1]]

// -----

// CHECK-LABEL: func @subview
func @subview(%arg0 : index, %arg1 : index, %arg2 : memref<?x?xf32>) {
  %0 = alloc() : memref<64x4xf32, offset: 0, strides: [4, 1]>
  %1 = subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
  to memref<?x?xf32, offset: ?, strides: [?, ?]>
  "linalg.copy"(%1, %arg2) :
    (memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32>) -> ()
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: subview
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @condBranchAlloca
func @condBranchAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloca() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: cond_br
//      CHECK: %[[ALLOCA:.*]] = alloca()
//      CHECK: br ^bb3(%[[ALLOCA:.*]])
// CHECK-NEXT: ^bb3
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: return

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElseAlloca
func @ifElseAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = alloca() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %5, %7 {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  "linalg.copy"(%7, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic
//      CHECK: %[[ALLOCA:.*]] = alloca()
// CHECK-NEXT: linalg.generic
//      CHECK: dealloc %[[ALLOC]]
//      CHECK: linalg.copy
// CHECK-NEXT: return

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElseNestedAlloca
func @ifElseNestedAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloca() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
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
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %7, %9 {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  "linalg.copy"(%9, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

// CHECK-NEXT: %[[ALLOCA:.*]] = alloca()
// CHECK-NEXT: linalg.generic
//      CHECK: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic
//      CHECK: linalg.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @nestedRegionsAndCondBranchAlloca
func @nestedRegionsAndCondBranchAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = alloca() : memref<2xf32>
    linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %1 {
    ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
      %tmp2 = exp %gen2_arg0 : f32
      linalg.yield %tmp2 : f32
    }: memref<2xf32>, memref<2xf32>
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
//      CHECK: (%[[cond:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %{{.*}}: {{.*}})
// CHECK-NEXT:   %[[ALLOC:.*]] = alloc()
// CHECK-NEXT:   cond_br %[[cond]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT:   linalg.generic {{{.*}}} %[[ARG1]], %[[ALLOC]]
//      CHECK:     %[[ALLOCA:.*]] = alloca()
// CHECK-NEXT:     linalg.generic {{{.*}}} %[[ARG1]], %[[ALLOCA]]
//      CHECK:     %{{.*}} = exp
//      CHECK:  ^[[BB3:.*]]({{.*}}):
//      CHECK:  linalg.copy
// CHECK-NEXT:  dealloc %[[ALLOC]]

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
