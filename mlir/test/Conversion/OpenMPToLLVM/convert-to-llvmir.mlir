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
    }) {operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 0, 0>} : (index, index, index, index, index, index) -> ()
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

// -----

// CHECK-LABEL: @_QPomp_target_data_region
// CHECK: (%[[ARG0:.*]]: !llvm.ptr<array<1024 x i32>>, %[[ARG1:.*]]: !llvm.ptr<i32>) {
// CHECK:   omp.target_data   map((tofrom -> %[[ARG0]] : !llvm.ptr<array<1024 x i32>>)) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           llvm.store %[[VAL_1]], %[[ARG1]] : !llvm.ptr<i32>
// CHECK:           omp.terminator
// CHECK:         }
// CHECK:         llvm.return

llvm.func @_QPomp_target_data_region(%a : !llvm.ptr<array<1024 x i32>>, %i : !llvm.ptr<i32>) {
  omp.target_data   map((tofrom -> %a : !llvm.ptr<array<1024 x i32>>)) {
    %1 = llvm.mlir.constant(10 : i32) : i32
    llvm.store %1, %i : !llvm.ptr<i32>
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @_QPomp_target(
// CHECK:                             %[[ARG_0:.*]]: !llvm.ptr<array<1024 x i32>>,
// CHECK:                             %[[ARG_1:.*]]: !llvm.ptr<i32>) {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           omp.target   thread_limit(%[[VAL_0]] : i32) map((tofrom -> %[[ARG_0]] : !llvm.ptr<array<1024 x i32>>)) {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:             llvm.store %[[VAL_1]], %[[ARG_1]] : !llvm.ptr<i32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

llvm.func @_QPomp_target(%a : !llvm.ptr<array<1024 x i32>>, %i : !llvm.ptr<i32>) {
  %0 = llvm.mlir.constant(64 : i32) : i32
  omp.target   thread_limit(%0 : i32) map((tofrom -> %a : !llvm.ptr<array<1024 x i32>>)) {
    %1 = llvm.mlir.constant(10 : i32) : i32
    llvm.store %1, %i : !llvm.ptr<i32>
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL: @_QPsb
// CHECK: omp.sections
// CHECK: omp.section
// CHECK: llvm.br
// CHECK: llvm.icmp
// CHECK: llvm.cond_br
// CHECK: llvm.br
// CHECK: omp.terminator
// CHECK: omp.terminator
// CHECK: llvm.return

llvm.func @_QPsb() {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.mlir.constant(10 : i64) : i64
  %2 = llvm.mlir.constant(1 : i64) : i64
  omp.sections   {
    omp.section {
      llvm.br ^bb1(%1 : i64)
    ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
      %4 = llvm.icmp "sgt" %3, %0 : i64
      llvm.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %5 = llvm.sub %3, %2  : i64
      llvm.br ^bb1(%5 : i64)
    ^bb3:  // pred: ^bb1
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK:  omp.reduction.declare @eqv_reduction : i32 init
// CHECK:  ^bb0(%{{.*}}: i32):
// CHECK:    %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK:    %[[TRUE_EXT:.*]] = llvm.zext %[[TRUE]] : i1 to i32
// CHECK:    omp.yield(%[[TRUE_EXT]] : i32)
// CHECK:  } combiner {
// CHECK:  ^bb0(%[[ARG_1:.*]]: i32, %[[ARG_2:.*]]: i32):
// CHECK:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:    %[[CMP_1:.*]] = llvm.icmp "ne" %[[ARG_1]], %[[ZERO]] : i32
// CHECK:    %[[CMP_2:.*]] = llvm.icmp "ne" %[[ARG_2]], %[[ZERO]] : i32
// CHECK:    %[[COMBINE_VAL:.*]] = llvm.icmp "eq" %[[CMP_1]], %[[CMP_2]] : i1
// CHECK:    %[[COMBINE_VAL_EXT:.*]] = llvm.zext %[[COMBINE_VAL]] : i1 to i32
// CHECK:    omp.yield(%[[COMBINE_VAL_EXT]] : i32)
// CHECK-LABEL:  @_QPsimple_reduction
// CHECK:    %[[RED_ACCUMULATOR:.*]] = llvm.alloca %{{.*}} x i32 {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"} : (i64) -> !llvm.ptr<i32>
// CHECK:    omp.parallel
// CHECK:      omp.wsloop reduction(@eqv_reduction -> %[[RED_ACCUMULATOR]] : !llvm.ptr<i32>) for
// CHECK:        omp.reduction %{{.*}}, %[[RED_ACCUMULATOR]] : i32, !llvm.ptr<i32>
// CHECK:        omp.yield
// CHECK:      omp.terminator
// CHECK:    llvm.return

omp.reduction.declare @eqv_reduction : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(true) : i1
  %1 = llvm.zext %0 : i1 to i32
  omp.yield(%1 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.mlir.constant(0 : i64) : i32
  %1 = llvm.icmp "ne" %arg0, %0 : i32
  %2 = llvm.icmp "ne" %arg1, %0 : i32
  %3 = llvm.icmp "eq" %1, %2 : i1
  %4 = llvm.zext %3 : i1 to i32
  omp.yield(%4 : i32)
}
llvm.func @_QPsimple_reduction(%arg0: !llvm.ptr<array<100 x i32>> {fir.bindc_name = "y"}) {
  %0 = llvm.mlir.constant(100 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(true) : i1
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.alloca %3 x i32 {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"} : (i64) -> !llvm.ptr<i32>
  %5 = llvm.zext %2 : i1 to i32
  llvm.store %5, %4 : !llvm.ptr<i32>
  omp.parallel   {
    %6 = llvm.alloca %3 x i32 {adapt.valuebyref, in_type = i32, operandSegmentSizes = array<i32: 0, 0>, pinned} : (i64) -> !llvm.ptr<i32>
    omp.wsloop   reduction(@eqv_reduction -> %4 : !llvm.ptr<i32>) for  (%arg1) : i32 = (%1) to (%0) inclusive step (%1) {
      llvm.store %arg1, %6 : !llvm.ptr<i32>
      %7 = llvm.load %6 : !llvm.ptr<i32>
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.sub %8, %3  : i64
      %10 = llvm.getelementptr %arg0[0, %9] : (!llvm.ptr<array<100 x i32>>, i64) -> !llvm.ptr<i32>
      %11 = llvm.load %10 : !llvm.ptr<i32>
      omp.reduction %11, %4 : i32, !llvm.ptr<i32>
      omp.yield
    }
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL:  @_QQmain
llvm.func @_QQmain() {
  %0 = llvm.mlir.constant(0 : index) : i64
  %1 = llvm.mlir.constant(5 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.alloca %3 x i32 : (i64) -> !llvm.ptr<i32>
// CHECK: omp.taskgroup
  omp.taskgroup   {
    %5 = llvm.trunc %2 : i64 to i32
    llvm.br ^bb1(%5, %1 : i32, i64)
  ^bb1(%6: i32, %7: i64):  // 2 preds: ^bb0, ^bb2
    %8 = llvm.icmp "sgt" %7, %0 : i64
    llvm.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.store %6, %4 : !llvm.ptr<i32>
// CHECK: omp.task
    omp.task   {
// CHECK: llvm.call @[[CALL_FUNC:.*]]({{.*}}) :
      llvm.call @_QFPdo_work(%4) : (!llvm.ptr<i32>) -> ()
// CHECK: omp.terminator
      omp.terminator
    }
    %9 = llvm.load %4 : !llvm.ptr<i32>
    %10 = llvm.add %9, %5  : i32
    %11 = llvm.sub %7, %2  : i64
    llvm.br ^bb1(%10, %11 : i32, i64)
  ^bb3:  // pred: ^bb1
    llvm.store %6, %4 : !llvm.ptr<i32>
// CHECK: omp.terminator
    omp.terminator
  }
  llvm.return
}
// CHECK: @[[CALL_FUNC]]
llvm.func @_QFPdo_work(%arg0: !llvm.ptr<i32> {fir.bindc_name = "i"}) {
  llvm.return
}
