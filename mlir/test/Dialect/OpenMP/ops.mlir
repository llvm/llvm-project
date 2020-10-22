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
// CHECK-SAME: ([[ARG0:%.*]]: !llvm.i32) {
func @omp_flush(%arg0 : !llvm.i32) -> () {
  // Test without data var
  // CHECK: omp.flush
  omp.flush

  // Test with one data var
  // CHECK: omp.flush([[ARG0]] : !llvm.i32)
  omp.flush(%arg0 : !llvm.i32)

  // Test with two data var
  // CHECK: omp.flush([[ARG0]], [[ARG0]] : !llvm.i32, !llvm.i32)
  omp.flush(%arg0, %arg0: !llvm.i32, !llvm.i32)

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
