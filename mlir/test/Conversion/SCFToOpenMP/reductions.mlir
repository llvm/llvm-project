// RUN: mlir-opt -convert-scf-to-openmp -split-input-file %s | FileCheck %s

// CHECK: omp.declare_reduction @[[$REDF:.*]] : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(0.000000e+00 : f32)
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[RES:.*]] = arith.addf %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK: atomic
// CHECK: ^{{.*}}(%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr):
// CHECK: %[[RHS:.*]] = llvm.load %[[ARG1]] : !llvm.ptr -> f32
// CHECK: llvm.atomicrmw fadd %[[ARG0]], %[[RHS]] monotonic

// CHECK-LABEL: @reduction1
func.func @reduction1(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  // CHECK: %[[CST:.*]] = arith.constant 0.0
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1
  // CHECK: %[[BUF:.*]] = llvm.alloca %[[ONE]] x f32
  // CHECK: llvm.store %[[CST]], %[[BUF]]
  %step = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  // CHECK: omp.parallel
  // CHECK: omp.wsloop
  // CHECK-SAME: reduction(@[[$REDF]] %[[BUF]] -> %[[PVT_BUF:[a-z0-9]+]]
  // CHECK: omp.loop_nest
  // CHECK: memref.alloca_scope
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%zero) -> (f32) {
    // CHECK: %[[CST_INNER:.*]] = arith.constant 1.0
    %one = arith.constant 1.0 : f32
    // CHECK: %[[PVT_VAL:.*]] = llvm.load %[[PVT_BUF]] : !llvm.ptr -> f32
    // CHECK: %[[ADD_RESULT:.*]] = arith.addf %[[PVT_VAL]], %[[CST_INNER]] : f32
    // CHECK: llvm.store %[[ADD_RESULT]], %[[PVT_BUF]] : f32, !llvm.ptr
    scf.reduce(%one : f32) {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = arith.addf %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
    // CHECK: omp.yield
  }
  // CHECK: omp.terminator
  // CHECK: llvm.load %[[BUF]]
  return
}

// -----

// Only check the declaration here, the rest is same as above.
// CHECK: omp.declare_reduction @{{.*}} : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(1.000000e+00 : f32)
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[RES:.*]] = arith.mulf %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK-NOT: atomic

// CHECK-LABEL: @reduction2
func.func @reduction2(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  %step = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%zero) -> (f32) {
    %one = arith.constant 1.0 : f32
    scf.reduce(%one : f32) {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
  }
  return
}

// -----

// Check the generation of declaration for arith.muli.
// Mostly, the same check as above, except for the types,
// the name of the op and the init value.

// CHECK: omp.declare_reduction @[[$REDI:.*]] : i32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: omp.yield(%[[INIT]] : i32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
// CHECK: %[[RES:.*]] = arith.muli %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : i32)

// CHECK-NOT: atomic

// CHECK-LABEL: @reduction_muli
func.func @reduction_muli(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  %step = arith.constant 1 : index
  %one = arith.constant 1 : i32
  // CHECK: %[[RED_VAR:.*]] = llvm.alloca %{{.*}} x i32 : (i64) -> !llvm.ptr
  // CHECK: omp.wsloop reduction(@[[$REDI]] %[[RED_VAR]] -> %[[RED_PVT_VAR:.*]] : !llvm.ptr)
  // CHECK: omp.loop_nest
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%one) -> (i32) {
    // CHECK: %[[C2:.*]] = arith.constant 2 : i32
    %pow2 = arith.constant 2 : i32
    // CHECK: %[[RED_PVT_VAL:.*]] = llvm.load %[[RED_PVT_VAR]] : !llvm.ptr -> i32
    // CHECK: %[[MUL_RESULT:.*]] = arith.muli %[[RED_PVT_VAL]], %[[C2]] : i32
    // CHECK: llvm.store %[[MUL_RESULT]], %[[RED_PVT_VAR]] : i32, !llvm.ptr
    scf.reduce(%pow2 : i32) {
    ^bb0(%lhs : i32, %rhs: i32):
      %res = arith.muli %lhs, %rhs : i32
      scf.reduce.return %res : i32
    }
  }
  return
}

// -----

// Only check the declaration here, the rest is same as above.
// CHECK: omp.declare_reduction @{{.*}} : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(-3.4
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[CMP:.*]] = arith.cmpf oge, %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK-NOT: atomic

// CHECK-LABEL: @reduction3
func.func @reduction3(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  %step = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%zero) -> (f32) {
    %one = arith.constant 1.0 : f32
    scf.reduce(%one : f32) {
    ^bb0(%lhs : f32, %rhs: f32):
      %cmp = arith.cmpf oge, %lhs, %rhs : f32
      %res = arith.select %cmp, %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
  }
  return
}

// -----

// CHECK: omp.declare_reduction @[[$REDF1:.*]] : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(-3.4
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[CMP:.*]] = arith.cmpf oge, %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK-NOT: atomic

// CHECK: omp.declare_reduction @[[$REDF2:.*]] : i64

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant
// CHECK: omp.yield(%[[INIT]] : i64)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64)
// CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[ARG1]], %[[ARG0]]
// CHECK: omp.yield(%[[RES]] : i64)

// CHECK: atomic
// CHECK: ^{{.*}}(%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr):
// CHECK: %[[RHS:.*]] = llvm.load %[[ARG1]] : !llvm.ptr -> i64
// CHECK: llvm.atomicrmw max %[[ARG0]], %[[RHS]] monotonic

// CHECK-LABEL: @reduction4
func.func @reduction4(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) -> (f32, i64) {
  %step = arith.constant 1 : index
  // CHECK: %[[ZERO:.*]] = arith.constant 0.0
  %zero = arith.constant 0.0 : f32
  // CHECK: %[[IONE:.*]] = arith.constant 1
  %ione = arith.constant 1 : i64
  // CHECK: %[[BUF1:.*]] = llvm.alloca %{{.*}} x f32
  // CHECK: llvm.store %[[ZERO]], %[[BUF1]]
  // CHECK: %[[BUF2:.*]] = llvm.alloca %{{.*}} x i64
  // CHECK: llvm.store %[[IONE]], %[[BUF2]]

  // CHECK: omp.parallel
  // CHECK: omp.wsloop
  // CHECK-SAME: reduction(@[[$REDF1]] %[[BUF1]] -> %[[PVT_BUF1:[a-z0-9]+]]
  // CHECK-SAME:           @[[$REDF2]] %[[BUF2]] -> %[[PVT_BUF2:[a-z0-9]+]]
  // CHECK: omp.loop_nest
  // CHECK: memref.alloca_scope
  %res:2 = scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                        step (%arg4, %step) init (%zero, %ione) -> (f32, i64) {
    // CHECK: %[[CST_ONE:.*]] = arith.constant 1.0{{.*}} : f32
    %one = arith.constant 1.0 : f32
    // CHECK: %[[CST_INT_ONE:.*]] = arith.fptosi
    %1 = arith.fptosi %one : f32 to i64
    // CHECK: %[[PVT_VAL1:.*]] = llvm.load %[[PVT_BUF1]] : !llvm.ptr -> f32
    // CHECK: %[[TEMP1:.*]] = arith.cmpf oge, %[[PVT_VAL1]], %[[CST_ONE]] : f32
    // CHECK: %[[CMP_VAL1:.*]] = arith.select %[[TEMP1]], %[[PVT_VAL1]], %[[CST_ONE]] : f32
    // CHECK: llvm.store %[[CMP_VAL1]], %[[PVT_BUF1]] : f32, !llvm.ptr
    // CHECK: %[[PVT_VAL2:.*]] = llvm.load %[[PVT_BUF2]] : !llvm.ptr -> i64
    // CHECK: %[[TEMP2:.*]] = arith.cmpi slt, %[[PVT_VAL2]], %[[CST_INT_ONE]] : i64
    // CHECK: %[[CMP_VAL2:.*]] = arith.select %[[TEMP2]], %[[CST_INT_ONE]], %[[PVT_VAL2]] : i64
    // CHECK: llvm.store %[[CMP_VAL2]], %[[PVT_BUF2]] : i64, !llvm.ptr
    scf.reduce(%one, %1 : f32, i64) {
    ^bb0(%lhs : f32, %rhs: f32):
      %cmp = arith.cmpf oge, %lhs, %rhs : f32
      %res = arith.select %cmp, %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }, {
    ^bb1(%lhs: i64, %rhs: i64):
      %cmp = arith.cmpi slt, %lhs, %rhs : i64
      %res = arith.select %cmp, %rhs, %lhs : i64
      scf.reduce.return %res : i64
    }
    // CHECK: omp.yield
  }
  // CHECK: omp.terminator
  // CHECK: %[[RES1:.*]] = llvm.load %[[BUF1]] : !llvm.ptr -> f32
  // CHECK: %[[RES2:.*]] = llvm.load %[[BUF2]] : !llvm.ptr -> i64
  // CHECK: return %[[RES1]], %[[RES2]]
  return %res#0, %res#1 : f32, i64
}
