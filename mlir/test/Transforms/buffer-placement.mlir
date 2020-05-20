// RUN: mlir-opt -buffer-placement -split-input-file %s | FileCheck %s -dump-input-on-failure

// This file checks the behaviour of BufferPlacement pass for moving Alloc and Dealloc
// operations and inserting the missing the DeallocOps in their correct positions.

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferPlacement Expected Behaviour: It should move the existing AllocOp to the entry block,
// and insert a DeallocOp at the exit block after CopyOp since %1 is an alias for %0 and %arg1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @condBranch
func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
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

// Test Case: Existing AllocOp with no users.
// BufferPlacement Expected Behaviour: It should insert a DeallocOp right before ReturnOp.

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
// BufferPlacement Expected Behaviour: It should move the existing AllocOp to the entry block
// and insert a DeallocOp at the exit block after CopyOp since %1 is an alias for %0 and %arg1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @criticalEdge
func @criticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
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
// BufferPlacement Expected Behaviour: It shouldn't move the alloc position. It only inserts
// a DeallocOp at the exit block after CopyOp since %1 is an alias for %0 and %arg1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @invCriticalEdge
func @invCriticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
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
// BufferPlacement Expected Behaviour: It shouldn't move the AllocOps. It only inserts two missing DeallocOps in the exit block.
// %5 is an alias for %0. Therefore, the DeallocOp for %0 should occur after the last GenericOp. The Dealloc for %7 should
// happen after the CopyOp.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElse
func @ifElse(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0, ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>), ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %5, %7 {
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
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: No users for buffer in if-else CFG
//    bb0 <- Initial position of AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3
// BufferPlacement Expected Behaviour: It shouldn't move the AllocOp. It only inserts a missing DeallocOp
// in the exit block since %5 or %6 are the latest aliases of %0.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @ifElseNoUsers
func @ifElseNoUsers(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0, ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>), ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  "linalg.copy"(%arg1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}

//      CHECK: dealloc
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
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  cond_br %arg0, ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>), ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
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
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %7, %9 {
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
// CHECK-NEXT: linalg.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// Test Case: Dead operations in a single block.
// BufferPlacement Expected Behaviour: It shouldn't move the AllocOps. It only inserts the two missing DeallocOps
// after the last GenericOp.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @redundantOperations
func @redundantOperations(%arg0: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  %1 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %0, %1 {
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
//                                            bb0
//                                           /   \
// Initial position of the first AllocOp -> bb1  bb2 <- Initial position of the second AllocOp
//                                           \   /
//                                            bb3
// BufferPlacement Expected Behaviour: Both AllocOps should be moved to the entry block. Both missing DeallocOps should be moved to
// the exit block after CopyOp since %arg2 is an alias for %0 and %1.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func @moving_alloc_and_inserting_missing_dealloc(%cond: i1, %arg0: memref<2xf32>, %arg1: memref<2xf32>){
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %0 {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }: memref<2xf32>, memref<2xf32>
  br ^exit(%0 : memref<2xf32>)
^bb2:
  %1 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %1 {
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

// Test Case: Invalid position of the DeallocOp. There is a user after deallocation.
//   bb0
//  /   \
// bb1  bb2 <- Initial position of AllocOp
//  \   /
//   bb3
// BufferPlacement Expected Behaviour: It should move the AllocOp to the entry block.
// The existing DeallocOp should be moved to exit block.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @moving_invalid_dealloc_op_complex
func @moving_invalid_dealloc_op_complex(%cond: i1, %arg0: memref<2xf32>, %arg1: memref<2xf32>){
  cond_br %cond, ^bb1, ^bb2
^bb1:
  br ^exit(%arg0 : memref<2xf32>)
^bb2:
  %1 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %1 {
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
func @inserting_missing_dealloc_simple(%arg0 : memref<2xf32>, %arg1: memref<2xf32>){
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %0 {
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

// Test Case: Moving invalid DeallocOp (there is a user after deallocation) in a single block.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @moving_invalid_dealloc_op
func @moving_invalid_dealloc_op(%arg0 : memref<2xf32>, %arg1: memref<2xf32>){
  %0 = alloc() : memref<2xf32>
  linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0, %0 {
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

