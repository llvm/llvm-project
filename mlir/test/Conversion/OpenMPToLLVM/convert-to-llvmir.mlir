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
    }) {operand_segment_sizes = dense<[2, 2, 2, 0, 0, 0, 0]> : vector<7xi32>} : (index, index, index, index, index, index) -> ()
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
  omp.atomic.read %b = %a memory_order(acquire) hint(contended) : !llvm.ptr<i32>
  return
}

// -----

// CHECK-LABEL: @threadprivate
// CHECK: (%[[ARG0:.*]]: !llvm.ptr<i32>)
// CHECK: %[[VAL0:.*]] = omp.threadprivate %[[ARG0]] : !llvm.ptr<i32> -> !llvm.ptr<i32>
func.func @threadprivate(%a: !llvm.ptr<i32>) -> () {
  %1 = omp.threadprivate %a : !llvm.ptr<i32> -> !llvm.ptr<i32>
  return
}
