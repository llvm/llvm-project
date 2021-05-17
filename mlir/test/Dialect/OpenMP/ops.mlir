// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func @omp_barrier() -> () {
  // CHECK: omp.barrier
  omp.barrier
  return
}

func @omp_master() -> () {
  // CHECK: omp.master
  omp.master {
    // CHECK: omp.terminator
    omp.terminator
  }

  return
}

func @omp_taskwait() -> () {
  // CHECK: omp.taskwait
  omp.taskwait
  return
}

func @omp_taskyield() -> () {
  // CHECK: omp.taskyield
  omp.taskyield
  return
}

// CHECK-LABEL: func @omp_flush
// CHECK-SAME: ([[ARG0:%.*]]: i32) {
func @omp_flush(%arg0 : i32) -> () {
  // Test without data var
  // CHECK: omp.flush
  omp.flush

  // Test with one data var
  // CHECK: omp.flush([[ARG0]] : i32)
  omp.flush(%arg0 : i32)

  // Test with two data var
  // CHECK: omp.flush([[ARG0]], [[ARG0]] : i32, i32)
  omp.flush(%arg0, %arg0: i32, i32)

  return
}

func @omp_terminator() -> () {
  // CHECK: omp.terminator
  omp.terminator
}

func @omp_parallel(%data_var : memref<i32>, %if_cond : i1, %num_threads : si32) -> () {
  // CHECK: omp.parallel if(%{{.*}}) num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.parallel" (%if_cond, %num_threads, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({

  // test without if condition
  // CHECK: omp.parallel num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
    "omp.parallel"(%num_threads, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({
      omp.terminator
    }) {operand_segment_sizes = dense<[0,1,1,1,1,1,1,1]>: vector<8xi32>, default_val = "defshared"} : (si32, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  // CHECK: omp.barrier
    omp.barrier

  // test without num_threads
  // CHECK: omp.parallel if(%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
    "omp.parallel"(%if_cond, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({
      omp.terminator
    }) {operand_segment_sizes = dense<[1,0,1,1,1,1,1,1]> : vector<8xi32>} : (i1, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  // test without allocate
  // CHECK: omp.parallel if(%{{.*}}) num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>)
    "omp.parallel"(%if_cond, %num_threads, %data_var, %data_var, %data_var, %data_var) ({
      omp.terminator
    }) {operand_segment_sizes = dense<[1,1,1,1,1,1,0,0]> : vector<8xi32>} : (i1, si32, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

    omp.terminator
  }) {operand_segment_sizes = dense<[1,1,1,1,1,1,1,1]> : vector<8xi32>, proc_bind_val = "spread"} : (i1, si32, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  // test with multiple parameters for single variadic argument
  // CHECK: omp.parallel private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>) allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.parallel" (%data_var, %data_var, %data_var, %data_var, %data_var, %data_var, %data_var) ({
    omp.terminator
  }) {operand_segment_sizes = dense<[0,0,1,2,1,1,1,1]> : vector<8xi32>} : (memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>, memref<i32>) -> ()

  return
}

func @omp_parallel_pretty(%data_var : memref<i32>, %if_cond : i1, %num_threads : si32, %allocator : si32) -> () {
 // CHECK: omp.parallel
 omp.parallel {
  omp.terminator
 }

 // CHECK: omp.parallel num_threads(%{{.*}} : si32)
 omp.parallel num_threads(%num_threads : si32) {
   omp.terminator
 }

 // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
 omp.parallel allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
   omp.terminator
 }

 // CHECK: omp.parallel private(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>)
 omp.parallel private(%data_var : memref<i32>, %data_var : memref<i32>) firstprivate(%data_var : memref<i32>) {
   omp.terminator
 }

 // CHECK omp.parallel shared(%{{.*}} : memref<i32>) copyin(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>)
 omp.parallel shared(%data_var : memref<i32>) copyin(%data_var : memref<i32>, %data_var : memref<i32>) {
   omp.parallel if(%if_cond: i1) {
     omp.terminator
   }
   omp.terminator
 }

 // CHECK omp.parallel if(%{{.*}}) num_threads(%{{.*}} : si32) private(%{{.*}} : memref<i32>) proc_bind(close)
 omp.parallel num_threads(%num_threads : si32) if(%if_cond: i1)
              private(%data_var : memref<i32>) proc_bind(close) {
   omp.terminator
 }

 return
}

// CHECK-LABEL: omp_wsloop
func @omp_wsloop(%lb : index, %ub : index, %step : index, %data_var : memref<i32>, %linear_var : i32, %chunk_var : i32) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>, %{{.*}} : memref<i32>) collapse(2) ordered(1)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %data_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,2,0,0,0,0,0]> : vector<9xi32>, collapse_val = 2, ordered_val = 1} :
    (index, index, index, memref<i32>, memref<i32>) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %linear_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,0,0,0,1,1,0]> : vector<9xi32>, schedule_val = "Static"} :
    (index, index, index, memref<i32>, i32) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) linear(%{{.*}} = %{{.*}} : memref<i32>, %{{.*}} = %{{.*}} : memref<i32>) schedule(static)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %data_var, %linear_var, %linear_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,0,0,0,2,2,0]> : vector<9xi32>, schedule_val = "Static"} :
    (index, index, index, memref<i32>, memref<i32>, i32, i32) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(dynamic = %{{.*}}) collapse(3) ordered(2)
  "omp.wsloop" (%lb, %ub, %step, %data_var, %data_var, %data_var, %data_var, %linear_var, %chunk_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,1,1,1,1,1,1]> : vector<9xi32>, schedule_val = "Dynamic", collapse_val = 3, ordered_val = 2} :
    (index, index, index, memref<i32>, memref<i32>, memref<i32>, memref<i32>, i32, i32) -> ()

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) schedule(auto) nowait
  "omp.wsloop" (%lb, %ub, %step, %data_var) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1,1,0,0,0,0,0]> : vector<9xi32>, nowait, schedule_val = "Auto"} :
    (index, index, index, memref<i32>) -> ()

  return
}

// CHECK-LABEL: omp_wsloop_pretty
func @omp_wsloop_pretty(%lb : index, %ub : index, %step : index,
                 %data_var : memref<i32>, %linear_var : i32, %chunk_var : i32) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) private(%data_var : memref<i32>) collapse(2) ordered(2) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) schedule(static) lastprivate(%data_var : memref<i32>) linear(%data_var = %linear_var : memref<i32>) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private(%{{.*}} : memref<i32>) firstprivate(%{{.*}} : memref<i32>) lastprivate(%{{.*}} : memref<i32>) linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static = %{{.*}}) collapse(3) ordered(2)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) ordered(2) private(%data_var : memref<i32>)
     firstprivate(%data_var : memref<i32>) lastprivate(%data_var : memref<i32>) linear(%data_var = %linear_var : memref<i32>)
     schedule(static = %chunk_var) collapse(3) {
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) private({{.*}} : memref<i32>)
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) private(%data_var : memref<i32>) {
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_wsloop_pretty_multi_block
func @omp_wsloop_pretty_multi_block(%lb : index, %ub : index, %step : index, %data1 : memref<?xi32>, %data2 : memref<?xi32>) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) {
    %1 = "test.payload"(%iv) : (index) -> (i32)
    br ^bb1(%1: i32)
  ^bb1(%arg: i32):
    memref.store %arg, %data1[%iv] : memref<?xi32>
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) {
    %c = "test.condition"(%iv) : (index) -> (i1)
    %v1 = "test.payload"(%iv) : (index) -> (i32)
    cond_br %c, ^bb1(%v1: i32), ^bb2(%v1: i32)
  ^bb1(%arg0: i32):
    memref.store %arg0, %data1[%iv] : memref<?xi32>
    br ^bb3
  ^bb2(%arg1: i32):
    memref.store %arg1, %data2[%iv] : memref<?xi32>
    br ^bb3
  ^bb3:
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step) {
    %c = "test.condition"(%iv) : (index) -> (i1)
    %v1 = "test.payload"(%iv) : (index) -> (i32)
    cond_br %c, ^bb1(%v1: i32), ^bb2(%v1: i32)
  ^bb1(%arg0: i32):
    memref.store %arg0, %data1[%iv] : memref<?xi32>
    omp.yield
  ^bb2(%arg1: i32):
    memref.store %arg1, %data2[%iv] : memref<?xi32>
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_wsloop_pretty_non_index
func @omp_wsloop_pretty_non_index(%lb1 : i32, %ub1 : i32, %step1 : i32, %lb2 : i64, %ub2 : i64, %step2 : i64,
                           %data1 : memref<?xi32>, %data2 : memref<?xi64>) -> () {

  // CHECK: omp.wsloop (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv1) : i32 = (%lb1) to (%ub1) step (%step1) {
    %1 = "test.payload"(%iv1) : (i32) -> (index)
    br ^bb1(%1: index)
  ^bb1(%arg1: index):
    memref.store %iv1, %data1[%arg1] : memref<?xi32>
    omp.yield
  }

  // CHECK: omp.wsloop (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
  omp.wsloop (%iv2) : i64 = (%lb2) to (%ub2) step (%step2) {
    %2 = "test.payload"(%iv2) : (i64) -> (index)
    br ^bb1(%2: index)
  ^bb1(%arg2: index):
    memref.store %iv2, %data2[%arg2] : memref<?xi64>
    omp.yield
  }

  return
}

// CHECK-LABEL: omp_wsloop_pretty_multiple
func @omp_wsloop_pretty_multiple(%lb1 : i32, %ub1 : i32, %step1 : i32, %lb2 : i32, %ub2 : i32, %step2 : i32, %data1 : memref<?xi32>) -> () {

  // CHECK: omp.wsloop (%{{.*}}, %{{.*}}) : i32 = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}})
  omp.wsloop (%iv1, %iv2) : i32 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %1 = "test.payload"(%iv1) : (i32) -> (index)
    %2 = "test.payload"(%iv2) : (i32) -> (index)
    memref.store %iv1, %data1[%1] : memref<?xi32>
    memref.store %iv2, %data1[%2] : memref<?xi32>
    omp.yield
  }

  return
}
