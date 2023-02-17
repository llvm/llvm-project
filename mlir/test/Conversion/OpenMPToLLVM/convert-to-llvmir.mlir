// RUN: mlir-opt -convert-openmp-to-llvm -split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @foo(i64, i64)
func.func private @foo(index, index)

// CHECK-LABEL: llvm.func @critical_block_arg
func.func @critical_block_arg() {
  // CHECK: omp.critical
  omp.critical {
  // CHECK-NEXT: ^[[BB0:.*]](%[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64):
  ^bb0(%arg1: index, %arg2: index):
    // CHECK-NEXT: llvm.call @foo(%[[ARG1]], %[[ARG2]]) : (i64, i64) -> ()
    func.call @foo(%arg1, %arg2) : (index, index) -> ()
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: llvm.func @master_block_arg
func.func @master_block_arg() {
  // CHECK: omp.master
  omp.master {
  // CHECK-NEXT: ^[[BB0:.*]](%[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64):
  ^bb0(%arg1: index, %arg2: index):
    // CHECK-DAG: %[[CAST_ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : i64 to index
    // CHECK-DAG: %[[CAST_ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : i64 to index
    // CHECK-NEXT: "test.payload"(%[[CAST_ARG1]], %[[CAST_ARG2]]) : (index, index) -> ()
    "test.payload"(%arg1, %arg2) : (index, index) -> ()
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: llvm.func @branch_loop
func.func @branch_loop() {
  %start = arith.constant 0 : index
  %end = arith.constant 0 : index
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK-NEXT: llvm.br ^[[BB1:.*]](%{{[0-9]+}}, %{{[0-9]+}} : i64, i64
    cf.br ^bb1(%start, %end : index, index)
  // CHECK-NEXT: ^[[BB1]](%[[ARG1:[0-9]+]]: i64, %[[ARG2:[0-9]+]]: i64):{{.*}}
  ^bb1(%0: index, %1: index):
    // CHECK-NEXT: %[[CMP:[0-9]+]] = llvm.icmp "slt" %[[ARG1]], %[[ARG2]] : i64
    %2 = arith.cmpi slt, %0, %1 : index
    // CHECK-NEXT: llvm.cond_br %[[CMP]], ^[[BB2:.*]](%{{[0-9]+}}, %{{[0-9]+}} : i64, i64), ^[[BB3:.*]]
    cf.cond_br %2, ^bb2(%end, %end : index, index), ^bb3
  // CHECK-NEXT: ^[[BB2]](%[[ARG3:[0-9]+]]: i64, %[[ARG4:[0-9]+]]: i64):
  ^bb2(%3: index, %4: index):
    // CHECK-NEXT: llvm.br ^[[BB1]](%[[ARG3]], %[[ARG4]] : i64, i64)
    cf.br ^bb1(%3, %4 : index, index)
  // CHECK-NEXT: ^[[BB3]]:
  ^bb3:
    omp.flush
    omp.barrier
    omp.taskwait
    omp.taskyield
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: @wsloop
// CHECK: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64)
func.func @wsloop(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK: omp.wsloop for (%[[ARG6:.*]], %[[ARG7:.*]]) : i64 = (%[[ARG0]], %[[ARG1]]) to (%[[ARG2]], %[[ARG3]]) step (%[[ARG4]], %[[ARG5]]) {
    "omp.wsloop"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) ({
    ^bb0(%arg6: index, %arg7: index):
      // CHECK-DAG: %[[CAST_ARG6:.*]] = builtin.unrealized_conversion_cast %[[ARG6]] : i64 to index
      // CHECK-DAG: %[[CAST_ARG7:.*]] = builtin.unrealized_conversion_cast %[[ARG7]] : i64 to index
      // CHECK: "test.payload"(%[[CAST_ARG6]], %[[CAST_ARG7]]) : (index, index) -> ()
      "test.payload"(%arg6, %arg7) : (index, index) -> ()
      omp.yield
    }) {operand_segment_sizes = array<i32: 2, 2, 2, 0, 0, 0, 0>} : (index, index, index, index, index, index) -> ()
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: @atomic_write
// CHECK: (%[[ARG0:.*]]: !llvm.ptr<i32>)
// CHECK: %[[VAL0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: omp.atomic.write %[[ARG0]] = %[[VAL0]] hint(none) memory_order(relaxed) : !llvm.ptr<i32>, i32
func.func @atomic_write(%a: !llvm.ptr<i32>) -> () {
  %1 = arith.constant 1 : i32
  omp.atomic.write %a = %1 hint(none) memory_order(relaxed) : !llvm.ptr<i32>, i32
  return
}

// -----

// CHECK-LABEL: @atomic_read
// CHECK: (%[[ARG0:.*]]: !llvm.ptr<i32>, %[[ARG1:.*]]: !llvm.ptr<i32>)
// CHECK: omp.atomic.read %[[ARG1]] = %[[ARG0]] memory_order(acquire) hint(contended) : !llvm.ptr<i32>
func.func @atomic_read(%a: !llvm.ptr<i32>, %b: !llvm.ptr<i32>) -> () {
  omp.atomic.read %b = %a memory_order(acquire) hint(contended) : !llvm.ptr<i32>, i32
  return
}

// -----

func.func @atomic_update() {
  %0 = llvm.mlir.addressof @_QFsEc : !llvm.ptr<i32>
  omp.atomic.update   %0 : !llvm.ptr<i32> {
  ^bb0(%arg0: i32):
    %1 = arith.constant 1 : i32
    %2 = arith.addi %arg0, %1  : i32
    omp.yield(%2 : i32)
  }
  return
}
llvm.mlir.global internal @_QFsEc() : i32 {
  %0 = arith.constant 10 : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @atomic_update
// CHECK: %[[GLOBAL_VAR:.*]] = llvm.mlir.addressof @_QFsEc : !llvm.ptr<i32>
// CHECK: omp.atomic.update   %[[GLOBAL_VAR]] : !llvm.ptr<i32> {
// CHECK: ^bb0(%[[IN_VAL:.*]]: i32):
// CHECK:   %[[CONST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   %[[OUT_VAL:.*]] = llvm.add %[[IN_VAL]], %[[CONST_1]]  : i32
// CHECK:   omp.yield(%[[OUT_VAL]] : i32)
// CHECK: }

// -----

// CHECK-LABEL: @threadprivate
// CHECK: (%[[ARG0:.*]]: !llvm.ptr<i32>)
// CHECK: %[[VAL0:.*]] = omp.threadprivate %[[ARG0]] : !llvm.ptr<i32> -> !llvm.ptr<i32>
func.func @threadprivate(%a: !llvm.ptr<i32>) -> () {
  %1 = omp.threadprivate %a : !llvm.ptr<i32> -> !llvm.ptr<i32>
  return
}

// -----

// CHECK:      llvm.func @simdloop_block_arg(%[[LOWER:.*]]: i32, %[[UPPER:.*]]: i32, %[[ITER:.*]]: i64) {
// CHECK:      omp.simdloop   for  (%[[ARG_0:.*]]) : i32 =
// CHECK-SAME:     (%[[LOWER]]) to (%[[UPPER]]) inclusive step (%[[LOWER]]) {
// CHECK:      llvm.br ^[[BB1:.*]](%[[ITER]] : i64)
// CHECK:        ^[[BB1]](%[[VAL_0:.*]]: i64):
// CHECK:          %[[VAL_1:.*]] = llvm.icmp "slt" %[[VAL_0]], %[[ITER]] : i64
// CHECK:          llvm.cond_br %[[VAL_1]], ^[[BB2:.*]], ^[[BB3:.*]]
// CHECK:        ^[[BB2]]:
// CHECK:          %[[VAL_2:.*]] = llvm.add %[[VAL_0]], %[[ITER]]  : i64
// CHECK:          llvm.br ^[[BB1]](%[[VAL_2]] : i64)
// CHECK:        ^[[BB3]]:
// CHECK:          omp.yield
func.func @simdloop_block_arg(%val : i32, %ub : i32, %i : index) {
  omp.simdloop   for  (%arg0) : i32 = (%val) to (%ub) inclusive step (%val) {
    cf.br ^bb1(%i : index)
  ^bb1(%0: index):
    %1 = arith.cmpi slt, %0, %i : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:
    %2 = arith.addi %0, %i : index
    cf.br ^bb1(%2 : index)
  ^bb3:
    omp.yield
  }
  return
}

// -----

// CHECK-LABEL: @task_depend
// CHECK:  (%[[ARG0:.*]]: !llvm.ptr<i32>) {
// CHECK:  omp.task depend(taskdependin -> %[[ARG0]] : !llvm.ptr<i32>) {
// CHECK:    omp.terminator
// CHECK:  }
// CHECK:   llvm.return
// CHECK: }

func.func @task_depend(%arg0: !llvm.ptr<i32>) {
  omp.task depend(taskdependin -> %arg0 : !llvm.ptr<i32>) {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: @_QPomp_target_data
// CHECK: (%[[ARG0:.*]]: !llvm.ptr<i32>, %[[ARG1:.*]]: !llvm.ptr<i32>, %[[ARG2:.*]]: !llvm.ptr<i32>, %[[ARG3:.*]]: !llvm.ptr<i32>)
// CHECK:         omp.target_enter_data   map((to -> %[[ARG0]] : !llvm.ptr<i32>), (to -> %[[ARG1]] : !llvm.ptr<i32>), (always, alloc -> %[[ARG2]] : !llvm.ptr<i32>))
// CHECK:         omp.target_exit_data   map((from -> %[[ARG0]] : !llvm.ptr<i32>), (from -> %[[ARG1]] : !llvm.ptr<i32>), (release -> %[[ARG2]] : !llvm.ptr<i32>), (always, delete -> %[[ARG3]] : !llvm.ptr<i32>))
// CHECK:         llvm.return

llvm.func @_QPomp_target_data(%a : !llvm.ptr<i32>, %b : !llvm.ptr<i32>, %c : !llvm.ptr<i32>, %d : !llvm.ptr<i32>) {
  omp.target_enter_data   map((to -> %a : !llvm.ptr<i32>), (to -> %b : !llvm.ptr<i32>), (always, alloc -> %c : !llvm.ptr<i32>))
  omp.target_exit_data   map((from -> %a : !llvm.ptr<i32>), (from -> %b : !llvm.ptr<i32>), (release -> %c : !llvm.ptr<i32>), (always, delete -> %d : !llvm.ptr<i32>))
  llvm.return
}
