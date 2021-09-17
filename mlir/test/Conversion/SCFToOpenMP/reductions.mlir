// RUN: mlir-opt -convert-scf-to-openmp -split-input-file %s | FileCheck %s

// CHECK: omp.reduction.declare @[[$REDF:.*]] : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(0.000000e+00 : f32)
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[RES:.*]] = addf %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK: atomic
// CHECK: ^{{.*}}(%[[ARG0:.*]]: !llvm.ptr<f32>, %[[ARG1:.*]]: !llvm.ptr<f32>):
// CHECK: %[[RHS:.*]] = llvm.load %[[ARG1]]
// CHECK: llvm.atomicrmw fadd %[[ARG0]], %[[RHS]] monotonic

// CHECK-LABEL: @reduction1
func @reduction1(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  // CHECK: %[[CST:.*]] = constant 0.0
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1
  // CHECK: llvm.intr.stacksave
  // CHECK: %[[BUF:.*]] = llvm.alloca %[[ONE]] x f32
  // CHECK: llvm.store %[[CST]], %[[BUF]]
  %step = constant 1 : index
  %zero = constant 0.0 : f32
  // CHECK: omp.parallel
  // CHECK: omp.wsloop
  // CHECK-SAME: reduction(@[[$REDF]] -> %[[BUF]]
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%zero) -> (f32) {
    // CHECK: %[[CST_INNER:.*]] = constant 1.0
    %one = constant 1.0 : f32
    // CHECK: omp.reduction %[[CST_INNER]], %[[BUF]]
    scf.reduce(%one) : f32 {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = addf %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
    // CHECK: omp.yield
  }
  // CHECK: omp.terminator
  // CHECK: llvm.load %[[BUF]]
  // CHECK: llvm.intr.stackrestore
  return
}

// -----

// Only check the declaration here, the rest is same as above.
// CHECK: omp.reduction.declare @{{.*}} : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(1.000000e+00 : f32)
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[RES:.*]] = mulf %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK-NOT: atomic

// CHECK-LABEL: @reduction2
func @reduction2(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  %step = constant 1 : index
  %zero = constant 0.0 : f32
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%zero) -> (f32) {
    %one = constant 1.0 : f32
    scf.reduce(%one) : f32 {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = mulf %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
  }
  return
}

// -----

// Only check the declaration here, the rest is same as above.
// CHECK: omp.reduction.declare @{{.*}} : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(-3.4
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[CMP:.*]] = cmpf oge, %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = select %[[CMP]], %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK-NOT: atomic

// CHECK-LABEL: @reduction3
func @reduction3(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) {
  %step = constant 1 : index
  %zero = constant 0.0 : f32
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                            step (%arg4, %step) init (%zero) -> (f32) {
    %one = constant 1.0 : f32
    scf.reduce(%one) : f32 {
    ^bb0(%lhs : f32, %rhs: f32):
      %cmp = cmpf oge, %lhs, %rhs : f32
      %res = select %cmp, %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
  }
  return
}

// -----

// CHECK: omp.reduction.declare @[[$REDF1:.*]] : f32

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant(-3.4
// CHECK: omp.yield(%[[INIT]] : f32)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK: %[[CMP:.*]] = cmpf oge, %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = select %[[CMP]], %[[ARG0]], %[[ARG1]]
// CHECK: omp.yield(%[[RES]] : f32)

// CHECK-NOT: atomic

// CHECK: omp.reduction.declare @[[$REDF2:.*]] : i64

// CHECK: init
// CHECK: %[[INIT:.*]] = llvm.mlir.constant
// CHECK: omp.yield(%[[INIT]] : i64)

// CHECK: combiner
// CHECK: ^{{.*}}(%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64)
// CHECK: %[[CMP:.*]] = cmpi slt, %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = select %[[CMP]], %[[ARG1]], %[[ARG0]]
// CHECK: omp.yield(%[[RES]] : i64)

// CHECK: atomic
// CHECK: ^{{.*}}(%[[ARG0:.*]]: !llvm.ptr<i64>, %[[ARG1:.*]]: !llvm.ptr<i64>):
// CHECK: %[[RHS:.*]] = llvm.load %[[ARG1]]
// CHECK: llvm.atomicrmw max %[[ARG0]], %[[RHS]] monotonic

// CHECK-LABEL: @reduction4
func @reduction4(%arg0 : index, %arg1 : index, %arg2 : index,
                 %arg3 : index, %arg4 : index) -> (f32, i64) {
  %step = constant 1 : index
  // CHECK: %[[ZERO:.*]] = constant 0.0
  %zero = constant 0.0 : f32
  // CHECK: %[[IONE:.*]] = constant 1
  %ione = constant 1 : i64
  // CHECK: %[[BUF1:.*]] = llvm.alloca %{{.*}} x f32
  // CHECK: llvm.store %[[ZERO]], %[[BUF1]]
  // CHECK: %[[BUF2:.*]] = llvm.alloca %{{.*}} x i64
  // CHECK: llvm.store %[[IONE]], %[[BUF2]]

  // CHECK: omp.parallel
  // CHECK: omp.wsloop
  // CHECK-SAME: reduction(@[[$REDF1]] -> %[[BUF1]]
  // CHECK-SAME:           @[[$REDF2]] -> %[[BUF2]]
  %res:2 = scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                        step (%arg4, %step) init (%zero, %ione) -> (f32, i64) {
    %one = constant 1.0 : f32
    // CHECK: omp.reduction %{{.*}}, %[[BUF1]]
    scf.reduce(%one) : f32 {
    ^bb0(%lhs : f32, %rhs: f32):
      %cmp = cmpf oge, %lhs, %rhs : f32
      %res = select %cmp, %lhs, %rhs : f32
      scf.reduce.return %res : f32
    }
    // CHECK: fptosi
    %1 = fptosi %one : f32 to i64
    // CHECK: omp.reduction %{{.*}}, %[[BUF2]]
    scf.reduce(%1) : i64 {
    ^bb1(%lhs: i64, %rhs: i64):
      %cmp = cmpi slt, %lhs, %rhs : i64
      %res = select %cmp, %rhs, %lhs : i64
      scf.reduce.return %res : i64
    }
    // CHECK: omp.yield
  }
  // CHECK: omp.terminator
  // CHECK: %[[RES1:.*]] = llvm.load %[[BUF1]]
  // CHECK: %[[RES2:.*]] = llvm.load %[[BUF2]]
  // CHECK: return %[[RES1]], %[[RES2]]
  return %res#0, %res#1 : f32, i64
}
