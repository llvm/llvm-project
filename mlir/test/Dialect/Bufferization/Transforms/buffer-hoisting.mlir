// RUN: mlir-opt -buffer-hoisting -split-input-file %s | FileCheck %s

// This file checks the behaviour of BufferHoisting pass for moving Alloc
// operations to their correct positions.

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferHoisting expected behavior: It should move the existing AllocOp to
// the entry block.

// CHECK-LABEL: func @condBranch
func.func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: cf.cond_br

// -----

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferHoisting expected behavior: It should not move the existing AllocOp
// to any other block since the alloc has a dynamic dependency to block argument
// %0 in bb2.

// CHECK-LABEL: func @condBranchDynamicType
func.func @condBranchDynamicType(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cf.cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  cf.br ^bb3(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = memref.alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  cf.br ^bb3(%1 : memref<?xf32>)
^bb3(%2: memref<?xf32>):
  test.copy(%2, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-NEXT: cf.cond_br
//      CHECK: ^bb2
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc(%[[IDX]])
// CHECK-NEXT: test.buffer_based

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
// BufferHoisting expected behavior: It should not move the existing AllocOp
// to any other block since the alloc has a dynamic dependency to block argument
// %0 in bb2.

// CHECK-LABEL: func @condBranchDynamicTypeNested
func.func @condBranchDynamicTypeNested(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cf.cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  cf.br ^bb6(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = memref.alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  cf.cond_br %arg0, ^bb3, ^bb4
^bb3:
  cf.br ^bb5(%1 : memref<?xf32>)
^bb4:
  cf.br ^bb5(%1 : memref<?xf32>)
^bb5(%2: memref<?xf32>):
  cf.br ^bb6(%2 : memref<?xf32>)
^bb6(%3: memref<?xf32>):
  cf.br ^bb7(%3 : memref<?xf32>)
^bb7(%4: memref<?xf32>):
  test.copy(%4, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-NEXT: cf.cond_br
//      CHECK: ^bb2
//      CHECK: ^bb2(%[[IDX:.*]]:{{.*}})
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc(%[[IDX]])
// CHECK-NEXT: test.buffer_based

// -----

// Test Case:
//    bb0
//   /   \
//  |    bb1 <- Initial position of AllocOp
//   \   /
//    bb2
// BufferHoisting expected behavior: It should move the existing AllocOp to
// the entry block.

// CHECK-LABEL: func @criticalEdge
func.func @criticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: cf.cond_br

// -----

// Test Case:
//    bb0 <- Initial position of the first AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3 <- Initial position of the second AllocOp
// BufferHoisting expected behavior: It shouldn't move the AllocOps.

// CHECK-LABEL: func @ifElse
func.func @ifElse(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%7: memref<2xf32>)
  test.copy(%7, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: cf.br ^bb3
//      CHECK: cf.br ^bb3
// CHECK-NEXT: ^bb3
//      CHECK: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: test.copy(%[[ALLOC1]]
// CHECK-NEXT: return

// -----

// Test Case: No users for buffer in if-else CFG
//    bb0 <- Initial position of AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3
// BufferHoisting expected behavior: It shouldn't move the AllocOp.

// CHECK-LABEL: func @ifElseNoUsers
func.func @ifElseNoUsers(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  test.copy(%arg1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based

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
// BufferHoisting expected behavior: AllocOps shouldn't be moved.

// CHECK-LABEL: func @ifElseNested
func.func @ifElseNested(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb5(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.cond_br %arg0, ^bb3(%3 : memref<2xf32>), ^bb4(%4 : memref<2xf32>)
^bb3(%5: memref<2xf32>):
  cf.br ^bb5(%5, %3 : memref<2xf32>, memref<2xf32>)
^bb4(%6: memref<2xf32>):
  cf.br ^bb5(%3, %6 : memref<2xf32>, memref<2xf32>)
^bb5(%7: memref<2xf32>, %8: memref<2xf32>):
  %9 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%9: memref<2xf32>)
  test.copy(%9, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: cf.br ^bb5
//      CHECK: cf.br ^bb5
//      CHECK: cf.br ^bb5
// CHECK-NEXT: ^bb5
//      CHECK: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based

// -----

// Test Case: Dead operations in a single block.
// BufferHoisting expected behavior: It shouldn't move the AllocOps.

// CHECK-LABEL: func @redundantOperations
func.func @redundantOperations(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based

// -----

// Test Case:
//                                     bb0
//                                    /   \
// Initial pos of the 1st AllocOp -> bb1  bb2 <- Initial pos of the 2nd AllocOp
//                                    \   /
//                                     bb3
// BufferHoisting expected behavior: Both AllocOps should be moved to the
// entry block.

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func.func @moving_alloc_and_inserting_missing_dealloc(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^exit(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  cf.br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  test.copy(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %{{.*}} = memref.alloc()
// CHECK-NEXT: %{{.*}} = memref.alloc()
// CHECK-NEXT: cf.cond_br

// -----

// Test Case: Invalid position of the DeallocOp. There is a user after
// deallocation.
//   bb0
//  /   \
// bb1  bb2 <- Initial position of AllocOp
//  \   /
//   bb3
// BufferHoisting expected behavior: It should move the AllocOp to the entry
// block.

// CHECK-LABEL: func @moving_invalid_dealloc_op_complex
func.func @moving_invalid_dealloc_op_complex(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^exit(%arg0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  memref.dealloc %1 : memref<2xf32>
  cf.br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  test.copy(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %{{.*}} = memref.alloc()
// CHECK-NEXT: cf.cond_br

// -----

// Test Case: Nested regions - This test defines a BufferBasedOp inside the
// region of a RegionBufferBasedOp.
// BufferHoisting expected behavior: The AllocOp for the BufferBasedOp should
// remain inside the region of the RegiobBufferBasedOp. The AllocOp of the
// RegionBufferBasedOp should be moved to the entry block.

// CHECK-LABEL: func @nested_regions_and_cond_branch
func.func @nested_regions_and_cond_branch(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = memref.alloc() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = math.exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}
// CHECK-NEXT:   %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT:   cf.cond_br
//      CHECK:   test.region_buffer_based
//      CHECK:     %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT:     test.buffer_based

// -----

// Test Case: nested region control flow
// The alloc position of %1 does not need to be changed and flows through
// both if branches until it is finally returned.

// CHECK-LABEL: func @nested_region_control_flow
func.func @nested_region_control_flow(
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
// CHECK-NEXT: %{{.*}} = scf.if
//      CHECK: else
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc(%arg0, %arg1)

// -----

// Test Case: nested region control flow with a nested buffer allocation in a
// divergent branch.
// The alloc positions of %1 does not need to be changed. %3 is moved upwards.

// CHECK-LABEL: func @nested_region_control_flow_div
func.func @nested_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.yield %3 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc(%arg0, %arg1)
// CHECK-NEXT: %{{.*}} = scf.if

// -----

// Test Case: deeply nested region control flow with a nested buffer allocation
// in a divergent branch.
// The alloc position of %1 does not need to be changed. Allocs %4 and %5 are
// moved upwards.

// CHECK-LABEL: func @nested_region_control_flow_div_nested
func.func @nested_region_control_flow_div_nested(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    %3 = scf.if %0 -> (memref<?x?xf32>) {
      scf.yield %1 : memref<?x?xf32>
    } else {
      %4 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
      scf.yield %4 : memref<?x?xf32>
    }
    scf.yield %3 : memref<?x?xf32>
  } else {
    %5 = memref.alloc(%arg1, %arg1) : memref<?x?xf32>
    scf.yield %5 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}
//      CHECK: %[[ALLOC0:.*]] = memref.alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc(%arg0, %arg1)
// CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc(%arg1, %arg1)
// CHECK-NEXT: %{{.*}} = scf.if

// -----

// Test Case: deeply nested region control flow with a nested buffer allocation
// that has dependency within a nested region should not be moved outside of
// this region.

// CHECK-LABEL: func @nested_region_control_flow_div_nested_dependencies
func.func @nested_region_control_flow_div_nested_dependencies(
  %arg0: i32,
  %arg1: i1,
  %arg2: index) -> memref<?x?xf32> {
  %0 = scf.if %arg1 -> (memref<?x?xf32>) {
    %1 = arith.constant 1 : i32
    %2 = arith.addi %arg0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = memref.alloc(%arg2, %3) : memref<?x?xf32>
    scf.yield %4 : memref<?x?xf32>
  } else {
    %1 = arith.constant 2 : i32
    %2 = arith.addi %arg0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = memref.alloc(%arg2, %3) : memref<?x?xf32>
    scf.yield %4 : memref<?x?xf32>
  }
  return %0 : memref<?x?xf32>
}

//      CHECK: (%[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %{{.*}} = scf.if
// CHECK-NEXT: %{{.*}} = arith.constant
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %[[FUNC:.*]] = arith.index_cast
// CHECK-NEXT: alloc(%arg2, %[[FUNC]])
// CHECK-NEXT: scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: %{{.*}} = arith.constant
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %[[FUNC:.*]] = arith.index_cast
// CHECK-NEXT: alloc(%arg2, %[[FUNC]])

// -----

// Test Case: nested region control flow within a region interface.
// The alloc positions of %0 does not need to be changed.

// CHECK-LABEL: func @inner_region_control_flow
func.func @inner_region_control_flow(%arg0 : index) -> memref<?x?xf32> {
  %0 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
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

//      CHECK: %[[ALLOC0:.*]] = memref.alloc(%arg0, %arg0)
// CHECK-NEXT: {{.*}} test.region_if

// -----

// Test Case: nested region control flow within a region interface including an
// allocation in a divergent branch.
// The alloc positions of %0 does not need to be changed. %2 is moved upwards.

// CHECK-LABEL: func @inner_region_control_flow_div
func.func @inner_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = test.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>) then {
    ^bb0(%arg2 : memref<?x?xf32>):
      test.region_if_yield %arg2 : memref<?x?xf32>
  } else {
    ^bb0(%arg2 : memref<?x?xf32>):
      %2 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
      test.region_if_yield %2 : memref<?x?xf32>
  } join {
    ^bb0(%arg2 : memref<?x?xf32>):
      test.region_if_yield %arg2 : memref<?x?xf32>
  }
  return %1 : memref<?x?xf32>
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc(%arg0, %arg0)
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc(%arg0, %arg1)
// CHECK-NEXT: {{.*}} test.region_if

// -----

// Test Case: Alloca operations shouldn't be moved.

// CHECK-LABEL: func @condBranchAlloca
func.func @condBranchAlloca(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloca() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: cf.cond_br
//      CHECK: ^bb2
//      CHECK: ^bb2
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based

// -----

// Test Case: Alloca operations shouldn't be moved. The alloc operation also
// shouldn't be moved analogously to the ifElseNested test.

// CHECK-LABEL: func @ifElseNestedAlloca
func.func @ifElseNestedAlloca(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  %0 = memref.alloca() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb5(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.cond_br %arg0, ^bb3(%3 : memref<2xf32>), ^bb4(%4 : memref<2xf32>)
^bb3(%5: memref<2xf32>):
  cf.br ^bb5(%5, %3 : memref<2xf32>, memref<2xf32>)
^bb4(%6: memref<2xf32>):
  cf.br ^bb5(%3, %6 : memref<2xf32>, memref<2xf32>)
^bb5(%7: memref<2xf32>, %8: memref<2xf32>):
  %9 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%9: memref<2xf32>)
  test.copy(%9, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based
//      CHECK: ^bb5
//      CHECK: ^bb5
//      CHECK: ^bb5
// CHECK-NEXT: ^bb5
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based

// -----

// Test Case: Alloca operations shouldn't be moved. The alloc operation should
// be moved in the beginning analogous to the nestedRegionsAndCondBranch test.

// CHECK-LABEL: func @nestedRegionsAndCondBranchAlloca
func.func @nestedRegionsAndCondBranchAlloca(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = memref.alloca() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = math.exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}
// CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT:   cf.cond_br
//      CHECK:   test.region_buffer_based
//      CHECK:     %[[ALLOCA:.*]] = memref.alloca()
// CHECK-NEXT:     test.buffer_based

// -----

// Test Case: structured control-flow loop using a nested alloc.
// The alloc positions of %3 will be moved upwards.

// CHECK-LABEL: func @loop_alloc
func.func @loop_alloc(
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

//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: {{.*}} scf.for
//      CHECK: %[[ALLOC1:.*]] = memref.alloc()

// -----

// Test Case: structured control-flow loop with a nested if operation using
// a deeply nested buffer allocation.
// The allocation %4 is not moved upwards.

// CHECK-LABEL: func @loop_nested_if_alloc
func.func @loop_nested_if_alloc(
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
// CHECK-NEXT: {{.*}} scf.for
//      CHECK: %[[ALLOC1:.*]] = memref.alloc()

// -----

// Test Case: several nested structured control-flow loops with a deeply nested
// buffer allocation inside an if operation.
// Same behavior is an loop_nested_if_alloc: The allocs are not moved upwards.

// CHECK-LABEL: func @loop_nested_alloc
func.func @loop_nested_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = scf.for %i2 = %lb to %ub step %step
      iter_args(%iterBuf2 = %iterBuf) -> memref<2xf32> {
      %3 = scf.for %i3 = %lb to %ub step %step
        iter_args(%iterBuf3 = %iterBuf2) -> memref<2xf32> {
        %4 = memref.alloc() : memref<2xf32>
        %5 = arith.cmpi eq, %i, %ub : index
        %6 = scf.if %5 -> (memref<2xf32>) {
          %7 = memref.alloc() : memref<2xf32>
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

//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
//      CHECK: %[[ALLOC2:.*]] = memref.alloc()

// -----

// CHECK-LABEL: func @loop_nested_alloc_dyn_dependency
func.func @loop_nested_alloc_dyn_dependency(
  %lb: index,
  %ub: index,
  %step: index,
  %arg0: index,
  %buf: memref<?xf32>,
  %res: memref<?xf32>) {
  %0 = memref.alloc(%arg0) : memref<?xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<?xf32> {
    %2 = scf.for %i2 = %lb to %ub step %step
      iter_args(%iterBuf2 = %iterBuf) -> memref<?xf32> {
      %3 = scf.for %i3 = %lb to %ub step %step
        iter_args(%iterBuf3 = %iterBuf2) -> memref<?xf32> {
        %5 = arith.cmpi eq, %i, %ub : index
        %6 = scf.if %5 -> (memref<?xf32>) {
          %7 = memref.alloc(%i3) : memref<?xf32>
          scf.yield %7 : memref<?xf32>
        } else {
          scf.yield %iterBuf3 : memref<?xf32>
        }
        scf.yield %6 : memref<?xf32>
      }
      scf.yield %3 : memref<?xf32>
    }
    scf.yield %0 : memref<?xf32>
  }
  test.copy(%1, %res) : (memref<?xf32>, memref<?xf32>)
  return
}


//      CHECK: %[[ALLOC0:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: {{.*}} = scf.for
//      CHECK: %[[ALLOC1:.*]] = memref.alloc({{.*}})
