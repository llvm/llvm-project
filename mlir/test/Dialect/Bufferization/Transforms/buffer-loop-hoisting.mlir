// RUN: mlir-opt -buffer-loop-hoisting -split-input-file %s | FileCheck %s

// This file checks the behavior of BufferLoopHoisting pass for moving Alloc
// operations in their correct positions.

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferLoopHoisting expected behavior: It should not move the AllocOp.

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

// CHECK-NEXT: cf.cond_br
//      CHECK: %[[ALLOC:.*]] = memref.alloc()

// -----

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferLoopHoisting expected behavior: It should not move the existing AllocOp
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

// Test Case: Nested regions - This test defines a BufferBasedOp inside the
// region of a RegionBufferBasedOp.
// BufferLoopHoisting expected behavior: The AllocOp for the BufferBasedOp
// should remain inside the region of the RegionBufferBasedOp. The AllocOp of
// the RegionBufferBasedOp should not be moved during this pass.

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
// CHECK-NEXT:   cf.cond_br
//      CHECK:   %[[ALLOC0:.*]] = memref.alloc()
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

// Test Case: structured control-flow loop using a nested alloc.
// The alloc positions of %3 should not be changed.

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
// The allocation %4 should not be moved upwards due to a back-edge dependency.

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

// Test Case: several nested structured control-flow loops with deeply nested
// buffer allocations inside an if operation.
// Behavior: The allocs %0, %4 and %9 are moved upwards, while %7 and %8 stay
// in their positions.

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
          %8 = memref.alloc() : memref<2xf32>
          scf.yield %8 : memref<2xf32>
        } else {
          scf.yield %iterBuf3 : memref<2xf32>
        }
        %9 = memref.alloc() : memref<2xf32>
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
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc()
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: {{.*}} = scf.for
//      CHECK: {{.*}} = scf.if
//      CHECK: %[[ALLOC3:.*]] = memref.alloc()
//      CHECK: %[[ALLOC4:.*]] = memref.alloc()

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
        %4 = memref.alloc(%i3) : memref<?xf32>
        %5 = arith.cmpi eq, %i, %ub : index
        %6 = scf.if %5 -> (memref<?xf32>) {
          %7 = memref.alloc(%i3) : memref<?xf32>
          scf.yield %7 : memref<?xf32>
        } else {
          scf.yield %iterBuf3 : memref<?xf32>
        }
        %8 = memref.alloc(%i3) : memref<?xf32>
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
//      CHECK: %[[ALLOC2:.*]] = memref.alloc({{.*}})

// -----

// CHECK-LABEL: func @hoist_one_loop
func.func @hoist_one_loop(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
      %2 = memref.alloc() : memref<2xf32>
      scf.yield %0 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: {{.*}} = scf.for

// -----

// CHECK-LABEL: func @no_hoist_one_loop
func.func @no_hoist_one_loop(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
      %1 = memref.alloc() : memref<2xf32>
      scf.yield %1 : memref<2xf32>
  }
  test.copy(%0, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: {{.*}} = scf.for
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc({{.*}})

// -----

// CHECK-LABEL: func @hoist_multiple_loop
func.func @hoist_multiple_loop(
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
        %3 = memref.alloc() : memref<2xf32>
        scf.yield %0 : memref<2xf32>
    }
    scf.yield %0 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: {{.*}} = scf.for

// -----

// CHECK-LABEL: func @no_hoist_one_loop_conditional
func.func @no_hoist_one_loop_conditional(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
      %1 = arith.cmpi eq, %i, %ub : index
      %2 = scf.if %1 -> (memref<2xf32>) {
        %3 = memref.alloc() : memref<2xf32>
        scf.yield %3 : memref<2xf32>
      } else {
        scf.yield %iterBuf : memref<2xf32>
      }
    scf.yield %2 : memref<2xf32>
  }
  test.copy(%0, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: {{.*}} = scf.for
//      CHECK: {{.*}} = scf.if
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc({{.*}})

// -----

// CHECK-LABEL: func @hoist_one_loop_conditional
func.func @hoist_one_loop_conditional(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = arith.cmpi eq, %lb, %ub : index
  %2 = scf.if %1 -> (memref<2xf32>) {
    %3 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
      %4 = memref.alloc() : memref<2xf32>
      scf.yield %0 : memref<2xf32>
    }
    scf.yield %0 : memref<2xf32>
  }
  else
  {
    scf.yield %0 : memref<2xf32>
  }
  test.copy(%2, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: {{.*}} = scf.if
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc({{.*}})
//      CHECK: {{.*}} = scf.for

// -----

// CHECK-LABEL: func @no_hoist_one_loop_dependency
func.func @no_hoist_one_loop_dependency(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
      %2 = memref.alloc(%i) : memref<?xf32>
      scf.yield %0 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc({{.*}})

// -----

// CHECK-LABEL: func @partial_hoist_multiple_loop_dependency
func.func @partial_hoist_multiple_loop_dependency(
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
        %3 = memref.alloc(%i) : memref<?xf32>
        scf.yield %0 : memref<2xf32>
    }
    scf.yield %0 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: {{.*}} = scf.for
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc({{.*}})
// CHECK-NEXT: {{.*}} = scf.for

// -----

// CHECK-LABEL: func @no_hoist_parallel
func.func @no_hoist_parallel(
    %lb: index,
    %ub: index,
    %step: index) {
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
      %0 = memref.alloc() : memref<2xf32>
      scf.reduce
  }
  return
}

//      CHECK: memref.alloc
// CHECK-NEXT: scf.reduce

// -----

func.func @no_hoist_forall(
    %lb: index,
    %ub: index,
    %step: index) {
  scf.forall (%i) = (%lb) to (%ub) step (%step) {
      %1 = memref.alloc() : memref<2xf32>
  }
  return
}

//      CHECK: scf.forall
// CHECK-NEXT: memref.alloc

// -----

// Test with allocas to ensure that op is also considered.

// CHECK-LABEL: func @hoist_alloca
func.func @hoist_alloca(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloca() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = scf.for %i2 = %lb to %ub step %step
      iter_args(%iterBuf2 = %iterBuf) -> memref<2xf32> {
        %3 = memref.alloca() : memref<2xf32>
        scf.yield %0 : memref<2xf32>
    }
    scf.yield %0 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOCA0:.*]] = memref.alloca({{.*}})
// CHECK-NEXT: %[[ALLOCA1:.*]] = memref.alloca({{.*}})
// CHECK-NEXT: {{.*}} = scf.for
