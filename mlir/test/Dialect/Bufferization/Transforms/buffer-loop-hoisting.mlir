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

// CHECK-LABEL: func @no_hoist_affine_parallel
func.func @no_hoist_affine_parallel(%out: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  affine.parallel (%i) = (0) to (2) {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %i, %buffer[%c0] : memref<1xindex>
    %value = memref.load %buffer[%c0] : memref<1xindex>
    memref.store %value, %out[%i] : memref<2xindex>
  }
  return
}

//  CHECK-NOT: memref.alloc
//      CHECK: affine.parallel
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[ALLOC]]
// CHECK-NEXT: memref.store %[[VALUE]]
//  CHECK-NOT: memref.alloc
//      CHECK: return
// CHECK-NEXT: }

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

// -----

// A nested entry block can be reachable even when its enclosing block is not.
// CHECK-LABEL: func @loop_unreachable_parent(
// CHECK: return
// CHECK: ^bb1:
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: scf.while
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
func.func @loop_unreachable_parent() {
  return
^dead:
  %c0 = arith.constant 0 : index
  %false = arith.constant false
  scf.while : () -> () {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %c0, %buffer[%c0] : memref<1xindex>
    scf.condition(%false)
  } do {
    scf.yield
  }
  return
}

// -----

// An exit alias does not carry the allocation back to before.
// CHECK-LABEL: func @while_exit_alias(
//      CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: %[[LAST:.*]] = scf.while
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
//      CHECK: scf.condition{{.*}} %[[ALLOC]]
//      CHECK: ^bb0(%[[CURRENT:.*]]: memref<1xindex>):
// CHECK-NEXT: %{{.*}} = memref.load %[[CURRENT]]
//      CHECK: memref.load %[[LAST]]
func.func @while_exit_alias(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %last = scf.while (%i = %c0) : (index) -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %i, %buffer[%c0] : memref<1xindex>
    %continue = arith.cmpi slt, %i, %n : index
    scf.condition(%continue) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    %value = memref.load %current[%c0] : memref<1xindex>
    %next = arith.addi %value, %c1 : index
    scf.yield %next : index
  }
  %result = memref.load %last[%c0] : memref<1xindex>
  return %result : index
}

// -----

// A loop-invariant allocation extent permits hoisting with an exit alias.
// CHECK-LABEL: func @while_exit_alias_invariant_extent(
// CHECK-SAME: %{{.*}}: i1, %[[SIZE:.*]]: index)
// CHECK: %[[ALLOC:.*]] = memref.alloc(%[[SIZE]])
// CHECK-NEXT: %{{.*}} = scf.while
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]]
func.func @while_exit_alias_invariant_extent(%condition: i1, %size: index) {
  %last = scf.while () : () -> memref<?xindex> {
    %buffer = memref.alloc(%size) : memref<?xindex>
    scf.condition(%condition) %buffer : memref<?xindex>
  } do {
  ^bb0(%current: memref<?xindex>):
    scf.yield
  }
  return
}

// -----

// An unrelated buffer may be carried to before while the allocation is hoisted.
// CHECK-LABEL: func @while_exit_alias_unrelated_carried(
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: %{{.*}}:2 = scf.while
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]], %{{.*}}
// CHECK: ^bb0(%{{.*}}: memref<1xindex>, %[[KEEP:.*]]: memref<1xindex>):
// CHECK-NEXT: scf.yield %[[KEEP]]
func.func @while_exit_alias_unrelated_carried(%condition: i1, %other: memref<1xindex>) {
  %last:2 = scf.while (%keep = %other)
      : (memref<1xindex>) -> (memref<1xindex>, memref<1xindex>) {
    %buffer = memref.alloc() : memref<1xindex>
    scf.condition(%condition) %buffer, %keep : memref<1xindex>, memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>, %keep: memref<1xindex>):
    scf.yield %keep : memref<1xindex>
  }
  return
}

// -----

// Casts and subviews of the loop result are included in the alias check.
// CHECK-LABEL: func @while_exit_alias_view(
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: %[[LAST:.*]] = scf.while
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK: scf.condition{{.*}} %[[ALLOC]]
// CHECK: %[[VIEW:.*]] = memref.subview %[[LAST]]
// CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[VIEW]]
// CHECK-NEXT: %{{.*}} = memref.load %[[CAST]]
func.func @while_exit_alias_view(%condition: i1) -> index {
  %c0 = arith.constant 0 : index
  %last = scf.while () : () -> memref<2xindex> {
    %buffer = memref.alloc() : memref<2xindex>
    memref.store %c0, %buffer[%c0] : memref<2xindex>
    scf.condition(%condition) %buffer : memref<2xindex>
  } do {
  ^bb0(%current: memref<2xindex>):
    scf.yield
  }
  %view = memref.subview %last[0] [1] [1] : memref<2xindex> to memref<1xindex>
  %cast = memref.cast %view : memref<1xindex> to memref<?xindex>
  %result = memref.load %cast[%c0] : memref<?xindex>
  return %result : index
}

// -----

// A view passed back to the allocation region prevents hoisting.
// CHECK-LABEL: func @no_hoist_while_carried_view(
//  CHECK-NOT: memref.alloc
//      CHECK: scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
//      CHECK: scf.condition{{.*}} %{{.*}}, %[[ALLOC]]
//      CHECK: ^bb0(%{{.*}}: index, %[[CURRENT:.*]]: memref<1xindex>):
//      CHECK: %[[CAST:.*]] = memref.cast %[[CURRENT]]
//      CHECK: %[[VIEW:.*]] = memref.subview %[[CAST]]
//      CHECK: scf.yield %{{.*}}, %[[VIEW]]
func.func @no_hoist_while_carried_view(%n: index, %seed: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.while (%i = %c0, %old = %seed)
      : (index, memref<1xindex>) -> (index, memref<1xindex>) {
    %fresh = memref.alloc() : memref<1xindex>
    memref.store %i, %fresh[%c0] : memref<1xindex>
    %previous = memref.load %old[%c0] : memref<1xindex>
    %continue = arith.cmpi slt, %i, %n : index
    scf.condition(%continue) %previous, %fresh : index, memref<1xindex>
  } do {
  ^bb0(%previous: index, %current: memref<1xindex>):
    %value = memref.load %current[%c0] : memref<1xindex>
    %next = arith.addi %value, %c1 : index
    %cast = memref.cast %current : memref<1xindex> to memref<?xindex>
    %view = memref.subview %cast[0] [1] [1] : memref<?xindex> to memref<1xindex>
    scf.yield %next, %view : index, memref<1xindex>
  }
  return %result#0 : index
}

// -----

// Uses of the loop result must also be checked for capture.
// CHECK-LABEL: func @no_hoist_while_exit_capture(
//  CHECK-NOT: memref.alloc
//      CHECK: %[[LAST:.*]] = scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
//      CHECK: scf.condition{{.*}} %[[ALLOC]]
//      CHECK: call @capture(%[[LAST]])
func.func private @capture(memref<1xindex>)
func.func @no_hoist_while_exit_capture(%condition: i1) {
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    scf.condition(%condition) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    scf.yield
  }
  func.call @capture(%last) : (memref<1xindex>) -> ()
  return
}

// -----

// A loop result used in a nested region is outside the supported alias uses.
// CHECK-LABEL: func @no_hoist_while_exit_nested_region(
// CHECK-NOT: memref.alloc
// CHECK: %[[LAST:.*]] = scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]]
// CHECK: scf.if
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[LAST]]
// CHECK-NEXT: scf.yield %[[VALUE]] : index
func.func @no_hoist_while_exit_nested_region(%condition: i1, %read: i1) -> index {
  %c0 = arith.constant 0 : index
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %c0, %buffer[%c0] : memref<1xindex>
    scf.condition(%condition) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    scf.yield
  }
  %result = scf.if %read -> (index) {
    %value = memref.load %last[%c0] : memref<1xindex>
    scf.yield %value : index
  } else {
    scf.yield %c0 : index
  }
  return %result : index
}

// -----

// Hoisting would require the loaded value to be invariant across iterations.
// CHECK-LABEL: func @no_hoist_while_invariant_load(
//  CHECK-NOT: memref.alloc
//      CHECK: scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
//      CHECK: scf.condition{{.*}} %[[ALLOC]]
//      CHECK: ^bb0(%[[CURRENT:.*]]: memref<1xindex>):
//      CHECK: memref.load %[[CURRENT]]{{.*}} invariant(true)
func.func @no_hoist_while_invariant_load(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %last = scf.while (%i = %c0) : (index) -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %i, %buffer[%c0] : memref<1xindex>
    %continue = arith.cmpi slt, %i, %n : index
    scf.condition(%continue) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    %value = memref.load %current[%c0] invariant(true) : memref<1xindex>
    %next = arith.addi %value, %c1 : index
    scf.yield %next : index
  }
  return
}

// -----

// An allocation freed in the loop cannot be reused by later iterations.
// CHECK-LABEL: func @no_hoist_while_dealloc(
//  CHECK-NOT: memref.alloc
//      CHECK: scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
//      CHECK: scf.condition{{.*}} %[[ALLOC]]
//      CHECK: ^bb0(%[[CURRENT:.*]]: memref<1xindex>):
// CHECK-NEXT: memref.dealloc %[[CURRENT]]
func.func @no_hoist_while_dealloc(%condition: i1) {
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    scf.condition(%condition) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    memref.dealloc %current : memref<1xindex>
    scf.yield
  }
  return
}

// -----

// A memory-effect-free pointer observation can expose allocation identity.
// CHECK-LABEL: func @no_hoist_while_pointer_escape(
// CHECK-NOT: memref.alloc
// CHECK: %[[LAST:.*]] = scf.while
// CHECK-NEXT: %[[BUF:.*]] = memref.alloc()
// CHECK-NEXT: memref.store {{.*}}, %[[BUF]]
// CHECK-NEXT: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BUF]]
// CHECK-NEXT: memref.store %[[ADDRESS]]
func.func @no_hoist_while_pointer_escape(%addresses: memref<2xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %last = scf.while (%i = %c0) : (index) -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %i, %buffer[%c0] : memref<1xindex>
    %address = memref.extract_aligned_pointer_as_index %buffer : memref<1xindex> -> index
    memref.store %address, %addresses[%i] : memref<2xindex>
    %continue = arith.cmpi slt, %i, %c1 : index
    scf.condition(%continue) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    %value = memref.load %current[%c0] : memref<1xindex>
    %next = arith.addi %value, %c1 : index
    scf.yield %next : index
  }
  %result = memref.load %last[%c0] : memref<1xindex>
  return %result : index
}

// -----

// Hoisting into acc.loop would allow a subsequent hoist to share the allocation
// between parallel iterations. Keep it inside scf.while.
// CHECK-LABEL: func @no_hoist_while_parallel_acc_parent(
//  CHECK-NOT: memref.alloc
//      CHECK: acc.parallel
// CHECK-NEXT: acc.loop
// CHECK-NEXT: %[[LAST:.*]] = scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
//      CHECK: scf.condition{{.*}} %[[ALLOC]]
//      CHECK: memref.load %[[LAST]]
func.func @no_hoist_while_parallel_acc_parent(%out: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %false = arith.constant false
  acc.parallel {
    acc.loop gang vector control(%i : index) = (%c0 : index)
        to (%c2 : index) step (%c1 : index) {
      %last = scf.while () : () -> memref<1xindex> {
        %buffer = memref.alloc() : memref<1xindex>
        memref.store %i, %buffer[%c0] : memref<1xindex>
        scf.condition(%false) %buffer : memref<1xindex>
      } do {
      ^bb0(%current: memref<1xindex>):
        scf.yield
      }
      %value = memref.load %last[%c0] : memref<1xindex>
      memref.store %value, %out[%i] : memref<2xindex>
      acc.yield
    } independent
    acc.yield
  }
  return
}

// -----

// Hoisting with exit aliases requires a loop directly in a function body,
// even when the enclosing loop is sequential.
// CHECK-LABEL: func @no_hoist_while_nested_for(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NOT: memref.alloc
// CHECK: %[[LAST:.*]] = scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK: scf.condition{{.*}} %[[ALLOC]]
// CHECK: memref.load %[[LAST]]
func.func @no_hoist_while_nested_for(%n: index, %condition: i1, %out: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %last = scf.while () : () -> memref<1xindex> {
      %buffer = memref.alloc() : memref<1xindex>
      memref.store %i, %buffer[%c0] : memref<1xindex>
      scf.condition(%condition) %buffer : memref<1xindex>
    } do {
    ^bb0(%current: memref<1xindex>):
      scf.yield
    }
    %value = memref.load %last[%c0] : memref<1xindex>
    memref.store %value, %out[%i] : memref<?xindex>
  }
  return
}

// -----

// A loop-variant allocation extent prevents hoisting even if the result only
// leaves through an exit edge.
// CHECK-LABEL: func @no_hoist_while_dynamic_size(
// CHECK-NOT: memref.alloc
// CHECK: %[[LAST:.*]] = scf.while
// CHECK-NEXT: %[[SIZE:.*]] = arith.addi
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[SIZE]])
// CHECK: scf.condition{{.*}} %[[ALLOC]]
// CHECK: memref.load %[[LAST]]
func.func @no_hoist_while_dynamic_size(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %last = scf.while (%i = %c0) : (index) -> memref<?xindex> {
    %size = arith.addi %i, %c1 : index
    %buffer = memref.alloc(%size) : memref<?xindex>
    memref.store %i, %buffer[%c0] : memref<?xindex>
    %continue = arith.cmpi slt, %i, %n : index
    scf.condition(%continue) %buffer : memref<?xindex>
  } do {
  ^bb0(%current: memref<?xindex>):
    %value = memref.load %current[%c0] : memref<?xindex>
    %next = arith.addi %value, %c1 : index
    scf.yield %next : index
  }
  %value = memref.load %last[%c0] : memref<?xindex>
  return %value : index
}

// -----

// A write to another buffer does not prevent the dominator-based hoist.
// CHECK-LABEL: func @hoist_with_unrelated_loop_effect(
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.store_with_a_loop_region
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[ALLOC]]
// CHECK-NEXT: memref.store %[[VALUE]]
func.func @hoist_with_unrelated_loop_effect(%out: memref<f32>, %value: f32) {
  test.store_with_a_loop_region %out <store_before_region = true> {
    %buffer = memref.alloc() : memref<f32>
    memref.store %value, %buffer[] : memref<f32>
    %loaded = memref.load %buffer[] : memref<f32>
    memref.store %loaded, %out[] : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  return
}

// -----

// Preserve dominator-based hoisting from sequential OpenACC loops.
// CHECK-LABEL: func @hoist_sequential_acc_loop(
//      CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: acc.loop
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[ALLOC]]
// CHECK-NEXT: memref.store %[[VALUE]]
func.func @hoist_sequential_acc_loop(%n: index, %out: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  acc.loop control(%i : index) = (%c0 : index) to (%n : index) step (%c1 : index) {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %i, %buffer[%c0] : memref<1xindex>
    %value = memref.load %buffer[%c0] : memref<1xindex>
    memref.store %value, %out[%i] : memref<?xindex>
    acc.yield
  } seq
  return
}

// -----

// A call may capture an allocation even when no alias is carried to before.
// CHECK-LABEL: func @no_hoist_while_capture(
// CHECK-NOT: memref.alloc
// CHECK: scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: call @capture(%[[ALLOC]])
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]]
func.func private @capture(memref<1xindex>)
func.func @no_hoist_while_capture(%condition: i1) {
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    func.call @capture(%buffer) : (memref<1xindex>) -> ()
    scf.condition(%condition) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    scf.yield
  }
  return
}

// -----

// The exit-alias fallback does not support stack allocations.
// CHECK-LABEL: func @no_hoist_while_alloca(
// CHECK-NOT: memref.alloca
// CHECK: scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloca()
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]]
func.func @no_hoist_while_alloca(%condition: i1) {
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloca() : memref<1xindex>
    scf.condition(%condition) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    scf.yield
  }
  return
}

// -----

// Hoist into the unreachable parent block, then stop the placement walk there.
// CHECK-LABEL: func @while_unreachable_parent(
// CHECK: return
// CHECK: ^bb1:
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: scf.while
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]]
func.func @while_unreachable_parent() {
  return
^dead:
  %false = arith.constant false
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    scf.condition(%false) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    scf.yield
  }
  return
}

// -----

// Storing an alias as a value can expose the buffer to later iterations.
// CHECK-LABEL: func @no_hoist_while_store_alias(
// CHECK-SAME: %{{.*}}: i1, %[[HOLDER:.*]]: memref<memref<1xindex>>
// CHECK-NOT: memref.alloc
// CHECK: scf.while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: scf.condition{{.*}} %[[ALLOC]]
// CHECK: ^bb0(%[[CURRENT:.*]]: memref<1xindex>):
// CHECK-NEXT: memref.store %[[CURRENT]], %[[HOLDER]][]
func.func @no_hoist_while_store_alias(%condition: i1, %holder: memref<memref<1xindex>>) {
  %last = scf.while () : () -> memref<1xindex> {
    %buffer = memref.alloc() : memref<1xindex>
    scf.condition(%condition) %buffer : memref<1xindex>
  } do {
  ^bb0(%current: memref<1xindex>):
    memref.store %current, %holder[] : memref<memref<1xindex>>
    scf.yield
  }
  return
}
