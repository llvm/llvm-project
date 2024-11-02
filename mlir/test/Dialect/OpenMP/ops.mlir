// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func.func @omp_barrier() -> () {
  // CHECK: omp.barrier
  omp.barrier
  return
}

func.func @omp_master() -> () {
  // CHECK: omp.master
  omp.master {
    // CHECK: omp.terminator
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_masked
func.func @omp_masked(%filtered_thread_id : i32) -> () {
  // CHECK: omp.masked filter(%{{.*}} : i32)
  "omp.masked" (%filtered_thread_id) ({
    omp.terminator
  }) : (i32) -> ()

  // CHECK: omp.masked
  "omp.masked" () ({
    omp.terminator
  }) : () -> ()
  return
}

func.func @omp_taskwait() -> () {
  // CHECK: omp.taskwait
  omp.taskwait
  return
}

func.func @omp_taskyield() -> () {
  // CHECK: omp.taskyield
  omp.taskyield
  return
}

// CHECK-LABEL: func @omp_flush
// CHECK-SAME: ([[ARG0:%.*]]: memref<i32>) {
func.func @omp_flush(%arg0 : memref<i32>) -> () {
  // Test without data var
  // CHECK: omp.flush
  omp.flush

  // Test with one data var
  // CHECK: omp.flush([[ARG0]] : memref<i32>)
  omp.flush(%arg0 : memref<i32>)

  // Test with two data var
  // CHECK: omp.flush([[ARG0]], [[ARG0]] : memref<i32>, memref<i32>)
  omp.flush(%arg0, %arg0: memref<i32>, memref<i32>)

  return
}

func.func @omp_terminator() -> () {
  // CHECK: omp.terminator
  omp.terminator
}

func.func @omp_parallel(%data_var : memref<i32>, %if_cond : i1, %num_threads : i32, %idx : index) -> () {
  // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>) if(%{{.*}}) num_threads(%{{.*}} : i32)
  "omp.parallel" (%data_var, %data_var, %if_cond, %num_threads) ({

  // test without if condition
  // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>) num_threads(%{{.*}} : i32)
    "omp.parallel"(%data_var, %data_var, %num_threads) ({
      omp.terminator
    }) {operandSegmentSizes = array<i32: 1,1,0,1,0,0>} : (memref<i32>, memref<i32>, i32) -> ()

  // CHECK: omp.barrier
    omp.barrier

  // test without num_threads
  // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>) if(%{{.*}})
    "omp.parallel"(%data_var, %data_var, %if_cond) ({
      omp.terminator
    }) {operandSegmentSizes = array<i32: 1,1,1,0,0,0>} : (memref<i32>, memref<i32>, i1) -> ()

  // test without allocate
  // CHECK: omp.parallel if(%{{.*}}) num_threads(%{{.*}} : i32)
    "omp.parallel"(%if_cond, %num_threads) ({
      omp.terminator
    }) {operandSegmentSizes = array<i32: 0,0,1,1,0,0>} : (i1, i32) -> ()

    omp.terminator
  }) {operandSegmentSizes = array<i32: 1,1,1,1,0,0>, proc_bind_kind = #omp<procbindkind spread>} : (memref<i32>, memref<i32>, i1, i32) -> ()

  // test with multiple parameters for single variadic argument
  // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.parallel" (%data_var, %data_var) ({
    omp.terminator
  }) {operandSegmentSizes = array<i32: 1,1,0,0,0,0>} : (memref<i32>, memref<i32>) -> ()

  // CHECK: omp.parallel
  omp.parallel {
    // CHECK-NOT: omp.terminator
    // CHECK: omp.distribute
    omp.distribute {
      // CHECK-NEXT: omp.wsloop
      omp.wsloop {
        // CHECK-NEXT: omp.loop_nest
        omp.loop_nest (%iv) : index = (%idx) to (%idx) step (%idx) {
          omp.yield
        }
        omp.terminator
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  } {omp.composite}

  // CHECK: omp.parallel
  omp.parallel {
    // CHECK-NOT: omp.terminator
    // CHECK: omp.distribute
    omp.distribute {
      // CHECK-NEXT: omp.wsloop
      omp.wsloop {
        // CHECK-NEXT: omp.simd
        omp.simd {
          // CHECK-NEXT: omp.loop_nest
          omp.loop_nest (%iv) : index = (%idx) to (%idx) step (%idx) {
            omp.yield
          }
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  } {omp.composite}

  return
}

func.func @omp_parallel_pretty(%data_var : memref<i32>, %if_cond : i1, %num_threads : i32, %allocator : si32) -> () {
 // CHECK: omp.parallel
 omp.parallel {
  omp.terminator
 }

 // CHECK: omp.parallel num_threads(%{{.*}} : i32)
 omp.parallel num_threads(%num_threads : i32) {
   omp.terminator
 }

 %n_index = arith.constant 2 : index
 // CHECK: omp.parallel num_threads(%{{.*}} : index)
 omp.parallel num_threads(%n_index : index) {
   omp.terminator
 }

 %n_i64 = arith.constant 4 : i64
 // CHECK: omp.parallel num_threads(%{{.*}} : i64)
 omp.parallel num_threads(%n_i64 : i64) {
   omp.terminator
 }

 // CHECK: omp.parallel allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
 omp.parallel allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
   omp.terminator
 }

 // CHECK: omp.parallel
 // CHECK-NEXT: omp.parallel if(%{{.*}})
 omp.parallel {
   omp.parallel if(%if_cond) {
     omp.terminator
   }
   omp.terminator
 }

 // CHECK: omp.parallel if(%{{.*}}) num_threads(%{{.*}} : i32) proc_bind(close)
 omp.parallel num_threads(%num_threads : i32) if(%if_cond) proc_bind(close) {
   omp.terminator
 }

  return
}

// CHECK-LABEL: omp_loop_nest
func.func @omp_loop_nest(%lb : index, %ub : index, %step : index) -> () {
  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    "omp.loop_nest" (%lb, %ub, %step) ({
    ^bb0(%iv: index):
      omp.yield
    }) : (index, index, index) -> ()
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}})
    "omp.loop_nest" (%lb, %ub, %step) ({
    ^bb0(%iv: index):
      omp.yield
    }) {loop_inclusive} : (index, index, index) -> ()
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}, %{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}})
    "omp.loop_nest" (%lb, %lb, %ub, %ub, %step, %step) ({
    ^bb0(%iv: index, %iv3: index):
      omp.yield
    }) : (index, index, index, index, index, index) -> ()
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    "omp.loop_nest" (%lb, %ub, %step) ({
    ^bb0(%iv: index):
      // CHECK: test.op1
      "test.op1"(%lb) : (index) -> ()
      // CHECK: test.op2
      "test.op2"() : () -> ()
      // CHECK: omp.yield
      omp.yield
    }) : (index, index, index) -> ()
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_loop_nest_pretty
func.func @omp_loop_nest_pretty(%lb : index, %ub : index, %step : index) -> () {
  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}})
    omp.loop_nest (%iv) : index = (%lb) to (%ub) inclusive step (%step) {
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}})
    omp.loop_nest (%iv1, %iv2) : index = (%lb, %lb) to (%ub, %ub) step (%step, %step) {
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest
    // CHECK-SAME: (%{{.*}}) : index =
    // CHECK-SAME: (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step)  {
      // CHECK: test.op1
      "test.op1"(%lb) : (index) -> ()
      // CHECK: test.op2
      "test.op2"() : () -> ()
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_loop_nest_pretty_multi_block
func.func @omp_loop_nest_pretty_multi_block(%lb : index, %ub : index,
    %step : index, %data1 : memref<?xi32>, %data2 : memref<?xi32>) -> () {

  omp.wsloop {
    // CHECK: omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %1 = "test.payload"(%iv) : (index) -> (i32)
      cf.br ^bb1(%1: i32)
    ^bb1(%arg: i32):
      memref.store %arg, %data1[%iv] : memref<?xi32>
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %c = "test.condition"(%iv) : (index) -> (i1)
      %v1 = "test.payload"(%iv) : (index) -> (i32)
      cf.cond_br %c, ^bb1(%v1: i32), ^bb2(%v1: i32)
    ^bb1(%arg0: i32):
      memref.store %arg0, %data1[%iv] : memref<?xi32>
      cf.br ^bb3
    ^bb2(%arg1: i32):
      memref.store %arg1, %data2[%iv] : memref<?xi32>
      cf.br ^bb3
    ^bb3:
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %c = "test.condition"(%iv) : (index) -> (i1)
      %v1 = "test.payload"(%iv) : (index) -> (i32)
      cf.cond_br %c, ^bb1(%v1: i32), ^bb2(%v1: i32)
    ^bb1(%arg0: i32):
      memref.store %arg0, %data1[%iv] : memref<?xi32>
      omp.yield
    ^bb2(%arg1: i32):
      memref.store %arg1, %data2[%iv] : memref<?xi32>
      omp.yield
    }
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_loop_nest_pretty_non_index
func.func @omp_loop_nest_pretty_non_index(%lb1 : i32, %ub1 : i32, %step1 : i32,
    %lb2 : i64, %ub2 : i64, %step2 : i64, %data1 : memref<?xi32>,
    %data2 : memref<?xi64>) -> () {

  omp.wsloop {
    // CHECK: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv1) : i32 = (%lb1) to (%ub1) step (%step1) {
      %1 = "test.payload"(%iv1) : (i32) -> (index)
      cf.br ^bb1(%1: index)
    ^bb1(%arg1: index):
      memref.store %iv1, %data1[%arg1] : memref<?xi32>
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop {
    // CHECK: omp.loop_nest (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
    omp.loop_nest (%iv) : i64 = (%lb2) to (%ub2) step (%step2) {
      %2 = "test.payload"(%iv) : (i64) -> (index)
      cf.br ^bb1(%2: index)
    ^bb1(%arg2: index):
      memref.store %iv, %data2[%arg2] : memref<?xi64>
      omp.yield
    }
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_loop_nest_pretty_multiple
func.func @omp_loop_nest_pretty_multiple(%lb1 : i32, %ub1 : i32, %step1 : i32,
    %lb2 : i32, %ub2 : i32, %step2 : i32, %data1 : memref<?xi32>) -> () {

  omp.wsloop {
    // CHECK: omp.loop_nest (%{{.*}}, %{{.*}}) : i32 = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}})
    omp.loop_nest (%iv1, %iv2) : i32 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
      %1 = "test.payload"(%iv1) : (i32) -> (index)
      %2 = "test.payload"(%iv2) : (i32) -> (index)
      memref.store %iv1, %data1[%1] : memref<?xi32>
      memref.store %iv2, %data1[%2] : memref<?xi32>
      omp.yield
    }
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_wsloop
func.func @omp_wsloop(%lb : index, %ub : index, %step : index, %data_var : memref<i32>, %linear_var : i32, %chunk_var : i32) -> () {

  // CHECK: omp.wsloop ordered(1) {
  // CHECK-NEXT: omp.loop_nest
  "omp.wsloop" () ({
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,0,0,0,0,0>, ordered = 1} :
    () -> ()

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static) {
  // CHECK-NEXT: omp.loop_nest
  "omp.wsloop" (%data_var, %linear_var) ({
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,1,1,0,0,0>, schedule_kind = #omp<schedulekind static>} :
    (memref<i32>, i32) -> ()

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>, %{{.*}} = %{{.*}} : memref<i32>) schedule(static) {
  // CHECK-NEXT: omp.loop_nest
  "omp.wsloop" (%data_var, %data_var, %linear_var, %linear_var) ({
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,2,2,0,0,0>, schedule_kind = #omp<schedulekind static>} :
    (memref<i32>, memref<i32>, i32, i32) -> ()

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>) ordered(2) schedule(dynamic = %{{.*}}) {
  // CHECK-NEXT: omp.loop_nest
  "omp.wsloop" (%data_var, %linear_var, %chunk_var) ({
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,1,1,0,0,1>, schedule_kind = #omp<schedulekind dynamic>, ordered = 2} :
    (memref<i32>, i32, i32) -> ()

  // CHECK: omp.wsloop nowait schedule(auto) {
  // CHECK-NEXT: omp.loop_nest
  "omp.wsloop" () ({
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,0,0,0,0,0>, nowait, schedule_kind = #omp<schedulekind auto>} :
    () -> ()

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.simd
  // CHECK-NEXT: omp.loop_nest
  "omp.wsloop" () ({
    omp.simd {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
      omp.terminator
    } {omp.composite}
    omp.terminator
  }) {omp.composite} : () -> ()

  return
}

// CHECK-LABEL: omp_wsloop_pretty
func.func @omp_wsloop_pretty(%lb : index, %ub : index, %step : index, %data_var : memref<i32>, %linear_var : i32, %chunk_var : i32, %chunk_var2 : i16) -> () {

  // CHECK: omp.wsloop ordered(2) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop ordered(2) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>) schedule(static) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop schedule(static) linear(%data_var = %linear_var : memref<i32>) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>) ordered(2) schedule(static = %{{.*}} : i32) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop ordered(2) linear(%data_var = %linear_var : memref<i32>) schedule(static = %chunk_var : i32) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>) ordered(2) schedule(dynamic = %{{.*}} : i32, nonmonotonic) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop ordered(2) linear(%data_var = %linear_var : memref<i32>) schedule(dynamic = %chunk_var : i32, nonmonotonic) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step)  {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop linear(%{{.*}} = %{{.*}} : memref<i32>) ordered(2) schedule(dynamic = %{{.*}} : i16, monotonic) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop ordered(2) linear(%data_var = %linear_var : memref<i32>) schedule(dynamic = %chunk_var2 : i16, monotonic) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop nowait {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop nowait {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop nowait order(concurrent) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop order(concurrent) nowait {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.wsloop nowait order(reproducible:concurrent) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop order(reproducible:concurrent) nowait {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.wsloop nowait order(unconstrained:concurrent) {
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop order(unconstrained:concurrent) nowait {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.simd
  // CHECK-NEXT: omp.loop_nest
  omp.wsloop {
    omp.simd {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
      omp.terminator
    } {omp.composite}
    omp.terminator
  } {omp.composite}

  return
}

// CHECK-LABEL: omp_simd
func.func @omp_simd(%lb : index, %ub : index, %step : index) -> () {
  // CHECK: omp.simd
  omp.simd {
    "omp.loop_nest" (%lb, %ub, %step) ({
    ^bb1(%iv2: index):
      "omp.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "omp.terminator"() : () -> ()
  }

  return
}

// CHECK-LABEL: omp_simd_aligned_list
func.func @omp_simd_aligned_list(%arg0 : index, %arg1 : index, %arg2 : index,
                                 %arg3 : memref<i32>, %arg4 : memref<i32>) -> () {
  // CHECK:      omp.simd aligned(
  // CHECK-SAME: %{{.*}} : memref<i32> -> 32 : i64,
  // CHECK-SAME: %{{.*}} : memref<i32> -> 128 : i64)
  "omp.simd"(%arg3, %arg4) ({
    "omp.loop_nest" (%arg0, %arg1, %arg2) ({
    ^bb1(%iv2: index):
      "omp.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "omp.terminator"() : () -> ()
  }) {alignments = [32, 128],
      operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>} : (memref<i32>, memref<i32>) -> ()
  return
}

// CHECK-LABEL: omp_simd_aligned_single
func.func @omp_simd_aligned_single(%arg0 : index, %arg1 : index, %arg2 : index,
                                   %arg3 : memref<i32>, %arg4 : memref<i32>) -> () {
  // CHECK: omp.simd aligned(%{{.*}} : memref<i32> -> 32 : i64)
  "omp.simd"(%arg3) ({
    "omp.loop_nest" (%arg0, %arg1, %arg2) ({
    ^bb1(%iv2: index):
      "omp.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "omp.terminator"() : () -> ()
  }) {alignments = [32],
      operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0>} : (memref<i32>) -> ()
  return
}

// CHECK-LABEL: omp_simd_nontemporal_list
func.func @omp_simd_nontemporal_list(%arg0 : index, %arg1 : index,
                                     %arg2 : index, %arg3 : memref<i32>,
                                     %arg4 : memref<i64>) -> () {
  // CHECK: omp.simd nontemporal(%{{.*}}, %{{.*}} : memref<i32>, memref<i64>)
  "omp.simd"(%arg3, %arg4) ({
    "omp.loop_nest" (%arg0, %arg1, %arg2) ({
    ^bb1(%iv2: index):
      "omp.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "omp.terminator"() : () -> ()
  }) {operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 0, 0>} : (memref<i32>, memref<i64>) -> ()
  return
}

// CHECK-LABEL: omp_simd_nontemporal_single
func.func @omp_simd_nontemporal_single(%arg0 : index, %arg1 : index,
                                       %arg2 : index, %arg3 : memref<i32>,
                                       %arg4 : memref<i64>) -> () {
  // CHECK: omp.simd nontemporal(%{{.*}} : memref<i32>)
  "omp.simd"(%arg3) ({
    "omp.loop_nest" (%arg0, %arg1, %arg2) ({
    ^bb1(%iv2: index):
      "omp.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "omp.terminator"() : () -> ()
  }) {operandSegmentSizes = array<i32: 0, 0, 0, 0, 1, 0, 0>} : (memref<i32>) -> ()
  return
}

// CHECK-LABEL: omp_simd_pretty
func.func @omp_simd_pretty(%lb : index, %ub : index, %step : index) -> () {
  // CHECK: omp.simd {
  omp.simd {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func @omp_simd_pretty_aligned(
func.func @omp_simd_pretty_aligned(%lb : index, %ub : index, %step : index,
                                   %data_var : memref<i32>,
                                   %data_var1 : memref<i32>) -> () {
  // CHECK:      omp.simd aligned(
  // CHECK-SAME: %{{.*}} : memref<i32> -> 32 : i64,
  // CHECK-SAME: %{{.*}} : memref<i32> -> 128 : i64)
  omp.simd aligned(%data_var :  memref<i32> -> 32, %data_var1 : memref<i32> -> 128) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_simd_pretty_if
func.func @omp_simd_pretty_if(%lb : index, %ub : index, %step : index, %if_cond : i1) -> () {
  // CHECK: omp.simd if(%{{.*}})
  omp.simd if(%if_cond) {
    omp.loop_nest (%iv): index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @omp_simd_pretty_nontemporal
func.func @omp_simd_pretty_nontemporal(%lb : index, %ub : index, %step : index,
                                       %data_var : memref<i32>,
                                       %data_var1 : memref<i32>) -> () {
  // CHECK: omp.simd nontemporal(%{{.*}}, %{{.*}} : memref<i32>, memref<i32>)
  omp.simd nontemporal(%data_var, %data_var1 : memref<i32>, memref<i32>) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_simd_pretty_order
func.func @omp_simd_pretty_order(%lb : index, %ub : index, %step : index) -> () {
  // CHECK: omp.simd order(concurrent)
  omp.simd order(concurrent) {
    omp.loop_nest (%iv): index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.simd order(reproducible:concurrent)
  omp.simd order(reproducible:concurrent) {
    omp.loop_nest (%iv): index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.simd order(unconstrained:concurrent)
  omp.simd order(unconstrained:concurrent) {
    omp.loop_nest (%iv): index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_simd_pretty_simdlen
func.func @omp_simd_pretty_simdlen(%lb : index, %ub : index, %step : index) -> () {
  // CHECK: omp.simd simdlen(2)
  omp.simd simdlen(2) {
    omp.loop_nest (%iv): index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_simd_pretty_safelen
func.func @omp_simd_pretty_safelen(%lb : index, %ub : index, %step : index) -> () {
  // CHECK: omp.simd safelen(2)
  omp.simd safelen(2) {
    omp.loop_nest (%iv): index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_distribute
func.func @omp_distribute(%chunk_size : i32, %data_var : memref<i32>, %arg0 : i32) -> () {
  // CHECK: omp.distribute
  "omp.distribute" () ({
    "omp.loop_nest" (%arg0, %arg0, %arg0) ({
    ^bb0(%iv: i32):
      "omp.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "omp.terminator"() : () -> ()
  }) {} : () -> ()
  // CHECK: omp.distribute
  omp.distribute {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute dist_schedule_static
  omp.distribute dist_schedule_static {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute dist_schedule_static dist_schedule_chunk_size(%{{.+}} : i32)
  omp.distribute dist_schedule_static dist_schedule_chunk_size(%chunk_size : i32) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute order(concurrent)
  omp.distribute order(concurrent) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute order(reproducible:concurrent)
  omp.distribute order(reproducible:concurrent) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute order(unconstrained:concurrent)
  omp.distribute order(unconstrained:concurrent) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute allocate(%{{.+}} : memref<i32> -> %{{.+}} : memref<i32>)
  omp.distribute allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
    omp.terminator
  }
  // CHECK: omp.distribute
  omp.distribute {
    omp.simd {
      omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
        omp.yield
      }
      omp.terminator
    } {omp.composite}
    omp.terminator
  } {omp.composite}
  return
}


// CHECK-LABEL: omp_target
func.func @omp_target(%if_cond : i1, %device : si32,  %num_threads : i32, %device_ptr: memref<i32>, %device_addr: memref<?xi32>, %map1: memref<?xi32>, %map2: memref<?xi32>) -> () {

    // Test with optional operands; if_expr, device, thread_limit, private, firstprivate and nowait.
    // CHECK: omp.target device({{.*}}) if({{.*}}) nowait thread_limit({{.*}})
    "omp.target"(%device, %if_cond, %num_threads) ({
       // CHECK: omp.terminator
       omp.terminator
    }) {nowait, operandSegmentSizes = array<i32: 0,0,0,1,0,1,0,0,0,0,1>} : ( si32, i1, i32 ) -> ()

    // Test with optional map clause.
    // CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[VAL_1:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(tofrom) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[VAL_2:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target has_device_addr(%[[VAL_5:.*]] : memref<?xi32>) is_device_ptr(%[[VAL_4:.*]] : memref<i32>) map_entries(%[[MAP_A]] -> {{.*}}, %[[MAP_B]] -> {{.*}} : memref<?xi32>, memref<?xi32>) {
    %mapv1 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(tofrom) capture(ByRef) -> memref<?xi32> {name = ""}
    %mapv2 = omp.map.info var_ptr(%map2 : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target is_device_ptr(%device_ptr : memref<i32>) has_device_addr(%device_addr : memref<?xi32>) map_entries(%mapv1 -> %arg0, %mapv2 -> %arg1 : memref<?xi32>, memref<?xi32>) {
      omp.terminator
    }
    // CHECK: %[[MAP_C:.*]] = omp.map.info var_ptr(%[[VAL_1:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(to) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: %[[MAP_D:.*]] = omp.map.info var_ptr(%[[VAL_2:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(always, from) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target map_entries(%[[MAP_C]] -> {{.*}}, %[[MAP_D]] -> {{.*}} : memref<?xi32>, memref<?xi32>) {
    %mapv3 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(to) capture(ByRef) -> memref<?xi32> {name = ""}
    %mapv4 = omp.map.info var_ptr(%map2 : memref<?xi32>, tensor<?xi32>)   map_clauses(always, from) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target map_entries(%mapv3 -> %arg0, %mapv4 -> %arg1 : memref<?xi32>, memref<?xi32>) {
      omp.terminator
    }
    // CHECK: omp.barrier
    omp.barrier

    return
}

func.func @omp_target_data (%if_cond : i1, %device : si32, %device_ptr: memref<i32>, %device_addr: memref<?xi32>, %map1: memref<?xi32>, %map2: memref<?xi32>) -> () {
    // CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[VAL_2:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(always, from) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target_data device(%[[VAL_1:.*]] : si32) if(%[[VAL_0:.*]]) map_entries(%[[MAP_A]] : memref<?xi32>)
    %mapv1 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(always, from) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target_data if(%if_cond) device(%device : si32) map_entries(%mapv1 : memref<?xi32>){}

    // CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[VAL_2:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(close, present, to) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target_data map_entries(%[[MAP_A]] : memref<?xi32>) use_device_addr(%[[VAL_3:.*]] -> %{{.*}} : memref<?xi32>) use_device_ptr(%[[VAL_4:.*]] -> %{{.*}} : memref<i32>)
    %mapv2 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(close, present, to) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target_data map_entries(%mapv2 : memref<?xi32>) use_device_addr(%device_addr -> %arg0 : memref<?xi32>) use_device_ptr(%device_ptr -> %arg1 : memref<i32>) {
      omp.terminator
    }

    // CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[VAL_1:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(tofrom) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[VAL_2:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target_data map_entries(%[[MAP_A]], %[[MAP_B]] : memref<?xi32>, memref<?xi32>)
    %mapv3 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(tofrom) capture(ByRef) -> memref<?xi32> {name = ""}
    %mapv4 = omp.map.info var_ptr(%map2 : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target_data map_entries(%mapv3, %mapv4 : memref<?xi32>, memref<?xi32>) {}

    // CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[VAL_3:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target_enter_data device(%[[VAL_1:.*]] : si32) if(%[[VAL_0:.*]]) map_entries(%[[MAP_A]] : memref<?xi32>) nowait
    %mapv5 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target_enter_data if(%if_cond) device(%device : si32) nowait map_entries(%mapv5 : memref<?xi32>)

    // CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[VAL_3:.*]] : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    // CHECK: omp.target_exit_data device(%[[VAL_1:.*]] : si32) if(%[[VAL_0:.*]]) map_entries(%[[MAP_A]] : memref<?xi32>) nowait
    %mapv6 = omp.map.info var_ptr(%map2 : memref<?xi32>, tensor<?xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}
    omp.target_exit_data if(%if_cond) device(%device : si32) nowait map_entries(%mapv6 : memref<?xi32>)

    return
}

// CHECK-LABEL: omp_target_pretty
func.func @omp_target_pretty(%if_cond : i1, %device : si32,  %num_threads : i32) -> () {
    // CHECK: omp.target device({{.*}}) if({{.*}})
    omp.target if(%if_cond) device(%device : si32) {
      omp.terminator
    }

    // CHECK: omp.target device({{.*}}) if({{.*}}) nowait
    omp.target if(%if_cond) device(%device : si32) thread_limit(%num_threads : i32) nowait {
      omp.terminator
    }

    return
}

// CHECK: omp.declare_reduction
// CHECK-LABEL: @add_f32
// CHECK: : f32
// CHECK: init
// CHECK: ^{{.+}}(%{{.+}}: f32):
// CHECK:   omp.yield
// CHECK: combiner
// CHECK: ^{{.+}}(%{{.+}}: f32, %{{.+}}: f32):
// CHECK:   omp.yield
// CHECK: atomic
// CHECK: ^{{.+}}(%{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr):
// CHECK:  omp.yield
// CHECK: cleanup
// CHECK:  omp.yield
omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
cleanup {
^bb0(%arg: f32):
  omp.yield
}

// CHECK-LABEL: func @wsloop_reduction
func.func @wsloop_reduction(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: reduction(@add_f32 %{{.+}} -> %[[PRV:.+]] : !llvm.ptr)
  omp.wsloop reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      // CHECK: %[[CST:.+]] = arith.constant 2.0{{.*}} : f32
      %cst = arith.constant 2.0 : f32
      // CHECK: %[[LPRV:.+]] = llvm.load %[[PRV]] : !llvm.ptr -> f32
      %lprv = llvm.load %prv : !llvm.ptr -> f32
      // CHECK: %[[RES:.+]] = llvm.fadd %[[LPRV]], %[[CST]] : f32
      %res = llvm.fadd %lprv, %cst: f32
      // CHECK: llvm.store %[[RES]], %[[PRV]] :  f32, !llvm.ptr
      llvm.store %res, %prv :  f32, !llvm.ptr
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @wsloop_reduction_byref
func.func @wsloop_reduction_byref(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: reduction(byref @add_f32 %{{.+}} -> %[[PRV:.+]] : !llvm.ptr)
  omp.wsloop reduction(byref @add_f32 %0 -> %prv : !llvm.ptr) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      // CHECK: %[[CST:.+]] = arith.constant 2.0{{.*}} : f32
      %cst = arith.constant 2.0 : f32
      // CHECK: %[[LPRV:.+]] = llvm.load %[[PRV]] : !llvm.ptr -> f32
      %lprv = llvm.load %prv : !llvm.ptr -> f32
      // CHECK: %[[RES:.+]] = llvm.fadd %[[LPRV]], %[[CST]] : f32
      %res = llvm.fadd %lprv, %cst: f32
      // CHECK: llvm.store %[[RES]], %[[PRV]] :  f32, !llvm.ptr
      llvm.store %res, %prv :  f32, !llvm.ptr
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @parallel_reduction
func.func @parallel_reduction() {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.parallel reduction(@add_f32 {{.+}} -> {{.+}} : !llvm.ptr)
  omp.parallel reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
    %1 = arith.constant 2.0 : f32
    %2 = llvm.load %prv : !llvm.ptr -> f32
    // CHECK: llvm.fadd %{{.*}}, %{{.*}} : f32
    %3 = llvm.fadd %1, %2 : f32
    llvm.store %3, %prv : f32, !llvm.ptr
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @parallel_reduction_byref
func.func @parallel_reduction_byref() {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.parallel reduction(byref @add_f32 {{.+}} -> {{.+}} : !llvm.ptr)
  omp.parallel reduction(byref @add_f32 %0 -> %prv : !llvm.ptr) {
    %1 = arith.constant 2.0 : f32
    %2 = llvm.load %prv : !llvm.ptr -> f32
    // CHECK: llvm.fadd %{{.*}}, %{{.*}} : f32
    %3 = llvm.fadd %1, %2 : f32
    llvm.store %3, %prv : f32, !llvm.ptr
    omp.terminator
  }
  return
}

// CHECK: func @parallel_wsloop_reduction
func.func @parallel_wsloop_reduction(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.parallel {
  omp.parallel {
    // CHECK: omp.wsloop reduction(@add_f32 %{{.*}} -> %{{.+}} : !llvm.ptr) {
    omp.wsloop reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
      // CHECK: omp.loop_nest (%{{.+}}) : index = (%{{.+}}) to (%{{.+}}) step (%{{.+}}) {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        %1 = arith.constant 2.0 : f32
        %2 = llvm.load %prv : !llvm.ptr -> f32
        // CHECK: llvm.fadd %{{.+}}, %{{.+}} : f32
        llvm.fadd %1, %2 : f32
        // CHECK: omp.yield
        omp.yield
      }
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_teams
func.func @omp_teams(%lb : i32, %ub : i32, %if_cond : i1, %num_threads : i32,
                     %data_var : memref<i32>) -> () {
  // Test nesting inside of omp.target
  omp.target {
    // CHECK: omp.teams
    omp.teams {
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.teams
  omp.teams {
    %0 = arith.constant 1 : i32
    // CHECK: omp.terminator
    omp.terminator
  }

  // Test num teams.
  // CHECK: omp.teams num_teams(%{{.+}} : i32 to %{{.+}} : i32)
  omp.teams num_teams(%lb : i32 to %ub : i32) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.teams num_teams( to %{{.+}} : i32)
  omp.teams num_teams(to %ub : i32) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // Test if.
  // CHECK: omp.teams if(%{{.+}})
  omp.teams if(%if_cond) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // Test thread limit.
  // CHECK: omp.teams thread_limit(%{{.+}} : i32)
  omp.teams thread_limit(%num_threads : i32) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // Test reduction.
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.teams reduction(@add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr) {
  omp.teams reduction(@add_f32 %0 -> %arg0 : !llvm.ptr) {
    %1 = arith.constant 2.0 : f32
    // CHECK: omp.terminator
    omp.terminator
  }

  // Test reduction byref
  // CHECK: omp.teams reduction(byref @add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr) {
  omp.teams reduction(byref @add_f32 %0 -> %arg0 : !llvm.ptr) {
    %1 = arith.constant 2.0 : f32
    // CHECK: omp.terminator
    omp.terminator
  }

  // Test allocate.
  // CHECK: omp.teams allocate(%{{.+}} : memref<i32> -> %{{.+}} : memref<i32>)
  omp.teams allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  return
}

// CHECK-LABEL: func @sections_reduction
func.func @sections_reduction() {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.sections reduction(@add_f32 %{{.+}} -> {{.+}} : !llvm.ptr)
  omp.sections reduction(@add_f32 %0 -> %arg0 : !llvm.ptr) {
    // CHECK: omp.section
    omp.section {
    ^bb0(%arg1 : !llvm.ptr):
      %1 = arith.constant 2.0 : f32
      omp.terminator
    }
    // CHECK: omp.section
    omp.section {
    ^bb0(%arg1 : !llvm.ptr):
      %1 = arith.constant 3.0 : f32
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @sections_reduction_byref
func.func @sections_reduction_byref() {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.sections reduction(byref @add_f32 %{{.+}} -> {{.+}} : !llvm.ptr)
  omp.sections reduction(byref @add_f32 %0 -> %arg0 : !llvm.ptr) {
    // CHECK: omp.section
    omp.section {
    ^bb0(%arg1 : !llvm.ptr):
      %1 = arith.constant 2.0 : f32
      omp.terminator
    }
    // CHECK: omp.section
    omp.section {
    ^bb0(%arg1 : !llvm.ptr):
      %1 = arith.constant 3.0 : f32
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK: omp.declare_reduction
// CHECK-LABEL: @add2_f32
omp.declare_reduction @add2_f32 : f32
// CHECK: init
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
// CHECK: combiner
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
// CHECK-NOT: atomic
// CHECK-NOT: cleanup

// CHECK-LABEL: func @wsloop_reduction2
func.func @wsloop_reduction2(%lb : index, %ub : index, %step : index) {
  %0 = memref.alloca() : memref<1xf32>
  // CHECK: omp.wsloop reduction(@add2_f32 %{{.+}} -> %{{.+}} : memref<1xf32>) {
  omp.wsloop reduction(@add2_f32 %0 -> %prv : memref<1xf32>) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %1 = arith.constant 2.0 : f32
      %2 = arith.constant 0 : index
      %3 = memref.load %prv[%2] : memref<1xf32>
      // CHECK: llvm.fadd
      %4 = llvm.fadd %1, %3 : f32
      memref.store %4, %prv[%2] : memref<1xf32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @parallel_reduction2
func.func @parallel_reduction2() {
  %0 = memref.alloca() : memref<1xf32>
  // CHECK: omp.parallel reduction(@add2_f32 %{{.+}} -> %{{.+}} : memref<1xf32>)
  omp.parallel reduction(@add2_f32 %0 -> %prv : memref<1xf32>) {
    %1 = arith.constant 2.0 : f32
    %2 = arith.constant 0 : index
    %3 = memref.load %prv[%2] : memref<1xf32>
    // CHECK: llvm.fadd
    %4 = llvm.fadd %1, %3 : f32
    memref.store %4, %prv[%2] : memref<1xf32>
    omp.terminator
  }
  return
}

// CHECK: func @parallel_wsloop_reduction2
func.func @parallel_wsloop_reduction2(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // CHECK: omp.parallel {
  omp.parallel {
    // CHECK: omp.wsloop reduction(@add2_f32 %{{.*}} -> %{{.+}} : !llvm.ptr) {
    omp.wsloop reduction(@add2_f32 %0 -> %prv : !llvm.ptr) {
      // CHECK: omp.loop_nest (%{{.+}}) : index = (%{{.+}}) to (%{{.+}}) step (%{{.+}}) {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        %1 = arith.constant 2.0 : f32
        %2 = llvm.load %prv : !llvm.ptr -> f32
        // CHECK: llvm.fadd %{{.+}}, %{{.+}} : f32
        %3 = llvm.fadd %1, %2 : f32
        // CHECK: omp.yield
        omp.yield
      }
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @sections_reduction2
func.func @sections_reduction2() {
  %0 = memref.alloca() : memref<1xf32>
  // CHECK: omp.sections reduction(@add2_f32 %{{.+}} -> %{{.+}} : memref<1xf32>)
  omp.sections reduction(@add2_f32 %0 -> %arg0 : memref<1xf32>) {
    omp.section {
    ^bb0(%arg1 : !llvm.ptr):
      %1 = arith.constant 2.0 : f32
      omp.terminator
    }
    omp.section {
    ^bb0(%arg1 : !llvm.ptr):
      %1 = arith.constant 2.0 : f32
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK: omp.critical.declare @mutex1 hint(uncontended)
omp.critical.declare @mutex1 hint(uncontended)
// CHECK: omp.critical.declare @mutex2 hint(contended)
omp.critical.declare @mutex2 hint(contended)
// CHECK: omp.critical.declare @mutex3 hint(nonspeculative)
omp.critical.declare @mutex3 hint(nonspeculative)
// CHECK: omp.critical.declare @mutex4 hint(speculative)
omp.critical.declare @mutex4 hint(speculative)
// CHECK: omp.critical.declare @mutex5 hint(uncontended, nonspeculative)
omp.critical.declare @mutex5 hint(uncontended, nonspeculative)
// CHECK: omp.critical.declare @mutex6 hint(contended, nonspeculative)
omp.critical.declare @mutex6 hint(contended, nonspeculative)
// CHECK: omp.critical.declare @mutex7 hint(uncontended, speculative)
omp.critical.declare @mutex7 hint(uncontended, speculative)
// CHECK: omp.critical.declare @mutex8 hint(contended, speculative)
omp.critical.declare @mutex8 hint(contended, speculative)
// CHECK: omp.critical.declare @mutex9
omp.critical.declare @mutex9 hint(none)
// CHECK: omp.critical.declare @mutex10
omp.critical.declare @mutex10


// CHECK-LABEL: omp_critical
func.func @omp_critical() -> () {
  // CHECK: omp.critical
  omp.critical {
    omp.terminator
  }

  // CHECK: omp.critical(@{{.*}})
  omp.critical(@mutex1) {
    omp.terminator
  }
  return
}

func.func @omp_ordered(%arg1 : i32, %arg2 : i32, %arg3 : i32,
    %vec0 : i64, %vec1 : i64, %vec2 : i64, %vec3 : i64) -> () {
  // CHECK: omp.ordered.region
  omp.ordered.region {
    // CHECK: omp.terminator
    omp.terminator
  }

  omp.wsloop ordered(0) {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3)  {
      // CHECK: omp.ordered.region
      omp.ordered.region {
        // CHECK: omp.terminator
        omp.terminator
      }
      omp.yield
    }
    omp.terminator
  }

  omp.wsloop ordered(1) {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // Only one DEPEND(SINK: vec) clause
      // CHECK: omp.ordered depend_type(dependsink) depend_vec(%{{.*}} : i64) {doacross_num_loops = 1 : i64}
      omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {doacross_num_loops = 1 : i64}

      // CHECK: omp.ordered depend_type(dependsource) depend_vec(%{{.*}} : i64) {doacross_num_loops = 1 : i64}
      omp.ordered depend_type(dependsource) depend_vec(%vec0 : i64) {doacross_num_loops = 1 : i64}

      omp.yield
    }
    omp.terminator
  }

  omp.wsloop ordered(2) {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // Multiple DEPEND(SINK: vec) clauses
      // CHECK: omp.ordered depend_type(dependsink) depend_vec(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i64, i64, i64, i64) {doacross_num_loops = 2 : i64}
      omp.ordered depend_type(dependsink) depend_vec(%vec0, %vec1, %vec2, %vec3 : i64, i64, i64, i64) {doacross_num_loops = 2 : i64}

      // CHECK: omp.ordered depend_type(dependsource) depend_vec(%{{.*}}, %{{.*}} : i64, i64) {doacross_num_loops = 2 : i64}
      omp.ordered depend_type(dependsource) depend_vec(%vec0, %vec1 : i64, i64) {doacross_num_loops = 2 : i64}

      omp.yield
    }
    omp.terminator
  }

  return
}

// CHECK-LABEL: omp_atomic_read
// CHECK-SAME: (%[[v:.*]]: memref<i32>, %[[x:.*]]: memref<i32>)
func.func @omp_atomic_read(%v: memref<i32>, %x: memref<i32>) {
  // CHECK: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  omp.atomic.read %v = %x : memref<i32>, i32
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(seq_cst) : memref<i32>, i32
  omp.atomic.read %v = %x memory_order(seq_cst) : memref<i32>, i32
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(acquire) : memref<i32>, i32
  omp.atomic.read %v = %x memory_order(acquire) : memref<i32>, i32
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(relaxed) : memref<i32>, i32
  omp.atomic.read %v = %x memory_order(relaxed) : memref<i32>, i32
  // CHECK: omp.atomic.read %[[v]] = %[[x]] hint(contended, nonspeculative) : memref<i32>, i32
  omp.atomic.read %v = %x hint(nonspeculative, contended) : memref<i32>, i32
  // CHECK: omp.atomic.read %[[v]] = %[[x]] hint(contended, speculative) memory_order(seq_cst) : memref<i32>, i32
  omp.atomic.read %v = %x hint(speculative, contended) memory_order(seq_cst) : memref<i32>, i32
  // CHECK: omp.atomic.read %[[v]] = %[[x]] memory_order(seq_cst) : memref<i32>, i32
  omp.atomic.read %v = %x hint(none) memory_order(seq_cst) : memref<i32>, i32
  return
}

// CHECK-LABEL: omp_atomic_write
// CHECK-SAME: (%[[ADDR:.*]]: memref<i32>, %[[VAL:.*]]: i32)
func.func @omp_atomic_write(%addr : memref<i32>, %val : i32) {
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] : memref<i32>, i32
  omp.atomic.write %addr = %val : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] memory_order(seq_cst) : memref<i32>, i32
  omp.atomic.write %addr = %val memory_order(seq_cst) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] memory_order(release) : memref<i32>, i32
  omp.atomic.write %addr = %val memory_order(release) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] memory_order(relaxed) : memref<i32>, i32
  omp.atomic.write %addr = %val memory_order(relaxed) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] hint(uncontended, speculative) : memref<i32>, i32
  omp.atomic.write %addr = %val hint(speculative, uncontended) : memref<i32>, i32
  // CHECK: omp.atomic.write %[[ADDR]] = %[[VAL]] : memref<i32>, i32
  omp.atomic.write %addr = %val hint(none) : memref<i32>, i32
  return
}

// CHECK-LABEL: omp_atomic_update
// CHECK-SAME: (%[[X:.*]]: memref<i32>, %[[EXPR:.*]]: i32, %[[XBOOL:.*]]: memref<i1>, %[[EXPRBOOL:.*]]: i1)
func.func @omp_atomic_update(%x : memref<i32>, %expr : i32, %xBool : memref<i1>, %exprBool : i1) {
  // CHECK: omp.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  // CHECK: omp.atomic.update %[[XBOOL]] : memref<i1>
  // CHECK-NEXT: (%[[XVAL:.*]]: i1):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.and %[[XVAL]], %[[EXPRBOOL]] : i1
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i1)
  omp.atomic.update %xBool : memref<i1> {
  ^bb0(%xval: i1):
    %newval = llvm.and %xval, %exprBool : i1
    omp.yield(%newval : i1)
  }
  // CHECK: omp.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.shl %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  // CHECK-NEXT: }
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.shl %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  // CHECK: omp.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.intr.smax(%[[XVAL]], %[[EXPR]]) : (i32, i32) -> i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  // CHECK-NEXT: }
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.intr.smax(%xval, %expr) : (i32, i32) -> i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update %[[XBOOL]] : memref<i1>
  // CHECK-NEXT: (%[[XVAL:.*]]: i1):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.icmp "eq" %[[XVAL]], %[[EXPRBOOL]] : i1
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i1)
  // }
  omp.atomic.update %xBool : memref<i1> {
  ^bb0(%xval: i1):
    %newval = llvm.icmp "eq" %xval, %exprBool : i1
    omp.yield(%newval : i1)
  }

  // CHECK: omp.atomic.update %[[X]] : memref<i32> {
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   omp.yield(%[[XVAL]] : i32)
  // CHECK-NEXT: }
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval:i32):
    omp.yield(%xval:i32)
  }

  // CHECK: omp.atomic.update %[[X]] : memref<i32> {
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   omp.yield(%{{.+}} : i32)
  // CHECK-NEXT: }
  %const = arith.constant 42 : i32
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval:i32):
    omp.yield(%const:i32)
  }

  // CHECK: omp.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(none) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(uncontended) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(uncontended) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(contended) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(contended) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(nonspeculative) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(nonspeculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(speculative) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(speculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(uncontended, nonspeculative) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(uncontended, nonspeculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(contended, nonspeculative) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(contended, nonspeculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(uncontended, speculative) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(uncontended, speculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(contended, speculative) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update hint(contended, speculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update memory_order(seq_cst) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update memory_order(seq_cst) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update memory_order(release) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update memory_order(release) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update memory_order(relaxed) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update memory_order(relaxed) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  // CHECK: omp.atomic.update hint(uncontended, speculative) memory_order(seq_cst) %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   omp.yield(%[[NEWVAL]] : i32)
  omp.atomic.update memory_order(seq_cst) hint(uncontended, speculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }

  return
}

// CHECK-LABEL: omp_atomic_capture
// CHECK-SAME: (%[[v:.*]]: memref<i32>, %[[x:.*]]: memref<i32>, %[[expr:.*]]: i32)
func.func @omp_atomic_capture(%v: memref<i32>, %x: memref<i32>, %expr: i32) {
  // CHECK: omp.atomic.capture {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture{
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }
  // CHECK: omp.atomic.capture {
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  omp.atomic.capture{
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }
  // CHECK: omp.atomic.capture {
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: omp.atomic.write %[[x]] = %[[expr]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture{
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.atomic.write %x = %expr : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(none) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(uncontended) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(uncontended) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(contended) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(contended) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(nonspeculative) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(nonspeculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(speculative) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(speculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(uncontended, nonspeculative) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(uncontended, nonspeculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(contended, nonspeculative) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(contended, nonspeculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(uncontended, speculative) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(uncontended, speculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(contended, speculative) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>
  // CHECK-NEXT: }
  omp.atomic.capture hint(contended, speculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture memory_order(seq_cst) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>
  // CHECK-NEXT: }
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture memory_order(acq_rel) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>
  // CHECK-NEXT: }
  omp.atomic.capture memory_order(acq_rel) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture memory_order(acquire) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture memory_order(acquire) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture memory_order(release) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture memory_order(release) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture memory_order(relaxed) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  // CHECK: omp.atomic.capture hint(contended, speculative) memory_order(seq_cst) {
  // CHECK-NEXT: omp.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   omp.yield(%[[newval]] : i32)
  // CHECK-NEXT: }
  // CHECK-NEXT: omp.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  omp.atomic.capture hint(contended, speculative) memory_order(seq_cst) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }

  return
}

// CHECK-LABEL: omp_sectionsop
func.func @omp_sectionsop(%data_var1 : memref<i32>, %data_var2 : memref<i32>,
                     %data_var3 : memref<i32>, %redn_var : !llvm.ptr) {
  // CHECK: omp.sections allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  "omp.sections" (%data_var1, %data_var1) ({
    // CHECK: omp.terminator
    omp.terminator
  }) {operandSegmentSizes = array<i32: 1,1,0,0>} : (memref<i32>, memref<i32>) -> ()

    // CHECK: omp.sections reduction(@add_f32 %{{.*}} -> %{{.*}} : !llvm.ptr)
  "omp.sections" (%redn_var) ({
  ^bb0(%arg0: !llvm.ptr):
    // CHECK: omp.terminator
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,0,1>, reduction_byref = array<i1: false>, reduction_syms=[@add_f32]} : (!llvm.ptr) -> ()

  // CHECK: omp.sections nowait {
  omp.sections nowait {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections reduction(@add_f32 %{{.*}} -> %{{.*}} : !llvm.ptr) {
  omp.sections reduction(@add_f32 %redn_var -> %arg0 : !llvm.ptr) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>)
  omp.sections allocate(%data_var1 : memref<i32> -> %data_var1 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.sections nowait
  omp.sections nowait {
    // CHECK: omp.section
    omp.section {
      // CHECK: %{{.*}} = "test.payload"() : () -> i32
      %1 = "test.payload"() : () -> i32
      // CHECK: %{{.*}} = "test.payload"() : () -> i32
      %2 = "test.payload"() : () -> i32
      // CHECK: %{{.*}} = "test.payload"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
      %3 = "test.payload"(%1, %2) : (i32, i32) -> i32
    }
    // CHECK: omp.section
    omp.section {
      // CHECK: %{{.*}} = "test.payload"(%{{.*}}) : (!llvm.ptr) -> i32
      %1 = "test.payload"(%redn_var) : (!llvm.ptr) -> i32
    }
    // CHECK: omp.section
    omp.section {
      // CHECK: "test.payload"(%{{.*}}) : (!llvm.ptr) -> ()
      "test.payload"(%redn_var) : (!llvm.ptr) -> ()
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @omp_single
func.func @omp_single() {
  omp.parallel {
    // CHECK: omp.single {
    omp.single {
      "test.payload"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @omp_single_nowait
func.func @omp_single_nowait() {
  omp.parallel {
    // CHECK: omp.single nowait {
    omp.single nowait {
      "test.payload"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @omp_single_allocate
func.func @omp_single_allocate(%data_var: memref<i32>) {
  omp.parallel {
    // CHECK: omp.single allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>) {
    omp.single allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
      "test.payload"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @omp_single_allocate_nowait
func.func @omp_single_allocate_nowait(%data_var: memref<i32>) {
  omp.parallel {
    // CHECK: omp.single allocate(%{{.*}} : memref<i32> -> %{{.*}} : memref<i32>) nowait {
    omp.single allocate(%data_var : memref<i32> -> %data_var : memref<i32>) nowait {
      "test.payload"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @omp_single_multiple_blocks
func.func @omp_single_multiple_blocks() {
  // CHECK: omp.single {
  omp.single {
    cf.br ^bb2
    ^bb2:
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

func.func private @copy_i32(memref<i32>, memref<i32>)

// CHECK-LABEL: func @omp_single_copyprivate
func.func @omp_single_copyprivate(%data_var: memref<i32>) {
  omp.parallel {
    // CHECK: omp.single copyprivate(%{{.*}} -> @copy_i32 : memref<i32>) {
    omp.single copyprivate(%data_var -> @copy_i32 : memref<i32>) {
      "test.payload"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_task
// CHECK-SAME: (%[[bool_var:.*]]: i1, %[[i64_var:.*]]: i64, %[[i32_var:.*]]: i32, %[[data_var:.*]]: memref<i32>)
func.func @omp_task(%bool_var: i1, %i64_var: i64, %i32_var: i32, %data_var: memref<i32>) {

  // Checking simple task
  // CHECK: omp.task {
  omp.task {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking `if` clause
  // CHECK: omp.task if(%[[bool_var]]) {
  omp.task if(%bool_var) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking `final` clause
  // CHECK: omp.task final(%[[bool_var]]) {
  omp.task final(%bool_var) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking `untied` clause
  // CHECK: omp.task untied {
  omp.task untied {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking `in_reduction` clause
  %c1 = arith.constant 1 : i32
  // CHECK: %[[redn_var1:.*]] = llvm.alloca %{{.*}} x f32 : (i32) -> !llvm.ptr
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  // CHECK: %[[redn_var2:.*]] = llvm.alloca %{{.*}} x f32 : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  // CHECK: omp.task in_reduction(@add_f32 %[[redn_var1]] -> %{{.+}}, @add_f32 %[[redn_var2]] -> %{{.+}} : !llvm.ptr, !llvm.ptr) {
  omp.task in_reduction(@add_f32 %0 -> %arg0, @add_f32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking `in_reduction` clause (mixed) byref
  // CHECK: omp.task in_reduction(byref @add_f32 %[[redn_var1]] -> %{{.+}}, @add_f32 %[[redn_var2]] -> %{{.+}} : !llvm.ptr, !llvm.ptr) {
  omp.task in_reduction(byref @add_f32 %0 -> %arg0, @add_f32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking priority clause
  // CHECK: omp.task priority(%[[i32_var]] : i32) {
  omp.task priority(%i32_var : i32) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking allocate clause
  // CHECK: omp.task allocate(%[[data_var]] : memref<i32> -> %[[data_var]] : memref<i32>) {
  omp.task allocate(%data_var : memref<i32> -> %data_var : memref<i32>) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  // Checking multiple clauses
  // CHECK: omp.task allocate(%[[data_var]] : memref<i32> -> %[[data_var]] : memref<i32>)
  omp.task allocate(%data_var : memref<i32> -> %data_var : memref<i32>)
      // CHECK-SAME: final(%[[bool_var]]) if(%[[bool_var]])
      final(%bool_var) if(%bool_var)
      // CHECK-SAME: priority(%[[i32_var]] : i32) untied
      priority(%i32_var : i32) untied
      // CHECK-SAME: in_reduction(@add_f32 %[[redn_var1]] -> %{{.+}}, byref @add_f32 %[[redn_var2]] -> %{{.+}} : !llvm.ptr, !llvm.ptr)
      in_reduction(@add_f32 %0 -> %arg0, byref @add_f32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }

  return
}

// CHECK-LABEL: @omp_task_depend
// CHECK-SAME: (%arg0: memref<i32>, %arg1: memref<i32>) {
func.func @omp_task_depend(%arg0: memref<i32>, %arg1: memref<i32>) {
  // CHECK:  omp.task   depend(taskdependin -> %arg0 : memref<i32>, taskdependin -> %arg1 : memref<i32>, taskdependinout -> %arg0 : memref<i32>) {
  omp.task   depend(taskdependin -> %arg0 : memref<i32>, taskdependin -> %arg1 : memref<i32>, taskdependinout -> %arg0 : memref<i32>) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}


// CHECK-LABEL: @omp_target_depend
// CHECK-SAME: (%arg0: memref<i32>, %arg1: memref<i32>) {
func.func @omp_target_depend(%arg0: memref<i32>, %arg1: memref<i32>) {
  // CHECK:  omp.target depend(taskdependin -> %arg0 : memref<i32>, taskdependin -> %arg1 : memref<i32>, taskdependinout -> %arg0 : memref<i32>) {
  omp.target depend(taskdependin -> %arg0 : memref<i32>, taskdependin -> %arg1 : memref<i32>, taskdependinout -> %arg0 : memref<i32>) {
    // CHECK: omp.terminator
    omp.terminator
  } {operandSegmentSizes = array<i32: 0,0,0,3,0,0,0,0>}
  return
}

func.func @omp_threadprivate() {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32

  // CHECK: [[ARG0:%.*]] = llvm.mlir.addressof @_QFsubEx : !llvm.ptr
  // CHECK: {{.*}} = omp.threadprivate [[ARG0]] : !llvm.ptr -> !llvm.ptr
  %3 = llvm.mlir.addressof @_QFsubEx : !llvm.ptr
  %4 = omp.threadprivate %3 : !llvm.ptr -> !llvm.ptr
  llvm.store %0, %4 : i32, !llvm.ptr

  // CHECK:  omp.parallel
  // CHECK:    {{.*}} = omp.threadprivate [[ARG0]] : !llvm.ptr -> !llvm.ptr
  omp.parallel  {
    %5 = omp.threadprivate %3 : !llvm.ptr -> !llvm.ptr
    llvm.store %1, %5 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.store %2, %4 : i32, !llvm.ptr
  return
}

llvm.mlir.global internal @_QFsubEx() : i32

func.func @omp_cancel_parallel(%if_cond : i1) -> () {
  // Test with optional operand; if_expr.
  omp.parallel {
    // CHECK: omp.cancel cancellation_construct_type(parallel) if(%{{.*}})
    omp.cancel cancellation_construct_type(parallel) if(%if_cond)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

func.func @omp_cancel_wsloop(%lb : index, %ub : index, %step : index) {
  omp.wsloop {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      // CHECK: omp.cancel cancellation_construct_type(loop)
      omp.cancel cancellation_construct_type(loop)
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }
  return
}

func.func @omp_cancel_sections() -> () {
  omp.sections {
    omp.section {
      // CHECK: omp.cancel cancellation_construct_type(sections)
      omp.cancel cancellation_construct_type(sections)
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

func.func @omp_cancellationpoint_parallel() -> () {
  omp.parallel {
    // CHECK: omp.cancellation_point cancellation_construct_type(parallel)
    omp.cancellation_point cancellation_construct_type(parallel)
    // CHECK: omp.cancel cancellation_construct_type(parallel)
    omp.cancel cancellation_construct_type(parallel)
    omp.terminator
  }
  return
}

func.func @omp_cancellationpoint_wsloop(%lb : index, %ub : index, %step : index) {
  omp.wsloop {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      // CHECK: omp.cancellation_point cancellation_construct_type(loop)
      omp.cancellation_point cancellation_construct_type(loop)
      // CHECK: omp.cancel cancellation_construct_type(loop)
      omp.cancel cancellation_construct_type(loop)
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }
  return
}

func.func @omp_cancellationpoint_sections() -> () {
  omp.sections {
    omp.section {
      // CHECK: omp.cancellation_point cancellation_construct_type(sections)
      omp.cancellation_point cancellation_construct_type(sections)
      // CHECK: omp.cancel cancellation_construct_type(sections)
      omp.cancel cancellation_construct_type(sections)
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_taskgroup_no_tasks
func.func @omp_taskgroup_no_tasks() -> () {

  // CHECK: omp.taskgroup
  omp.taskgroup {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_taskgroup_multiple_tasks
func.func @omp_taskgroup_multiple_tasks() -> () {
  // CHECK: omp.taskgroup
  omp.taskgroup {
    // CHECK: omp.task
    omp.task {
      "test.foo"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.task
    omp.task {
      "test.foo"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_taskgroup_clauses
func.func @omp_taskgroup_clauses() -> () {
  %testmemref = "test.memref"() : () -> (memref<i32>)
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  // CHECK: omp.taskgroup allocate(%{{.+}}: memref<i32> -> %{{.+}} : memref<i32>) task_reduction(@add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr)
  omp.taskgroup allocate(%testmemref : memref<i32> -> %testmemref : memref<i32>) task_reduction(@add_f32 %testf32 -> %arg0 : !llvm.ptr) {
    // CHECK: omp.task
    omp.task {
      "test.foo"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.task
    omp.task {
      "test.foo"() : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_taskloop
func.func @omp_taskloop(%lb: i32, %ub: i32, %step: i32) -> () {

  // CHECK: omp.taskloop {
  omp.taskloop {
    omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%step)  {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  %testbool = "test.bool"() : () -> (i1)

  // CHECK: omp.taskloop if(%{{[^)]+}}) {
  omp.taskloop if(%testbool) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop final(%{{[^)]+}}) {
  omp.taskloop final(%testbool) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop untied {
  omp.taskloop untied {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop mergeable {
  omp.taskloop mergeable {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // CHECK: omp.taskloop in_reduction(@add_f32 %{{.+}} -> %{{.+}}, @add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr, !llvm.ptr) {
  omp.taskloop in_reduction(@add_f32 %testf32 -> %arg0, @add_f32 %testf32_2 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // Checking byref attribute for in_reduction
  // CHECK: omp.taskloop in_reduction(byref @add_f32 %{{.+}} -> %{{.+}}, @add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr, !llvm.ptr) {
  omp.taskloop in_reduction(byref @add_f32 %testf32 -> %arg0, @add_f32 %testf32_2 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop reduction(byref @add_f32 %{{.+}} -> %{{.+}}, @add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr, !llvm.ptr) {
  omp.taskloop reduction(byref @add_f32 %testf32 -> %arg0, @add_f32 %testf32_2 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // check byref attrbute for reduction
  // CHECK: omp.taskloop reduction(byref @add_f32 %{{.+}} -> %{{.+}}, byref @add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr, !llvm.ptr) {
  omp.taskloop reduction(byref @add_f32 %testf32 -> %arg0, byref @add_f32 %testf32_2 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop in_reduction(@add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr) reduction(@add_f32 %{{.+}} -> %{{.+}} : !llvm.ptr) {
  omp.taskloop in_reduction(@add_f32 %testf32 -> %arg0 : !llvm.ptr) reduction(@add_f32 %testf32_2 -> %arg1 : !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  %testi32 = "test.i32"() : () -> (i32)
  // CHECK: omp.taskloop priority(%{{[^:]+}}: i32) {
  omp.taskloop priority(%testi32 : i32) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  %testmemref = "test.memref"() : () -> (memref<i32>)
  // CHECK: omp.taskloop allocate(%{{.+}} : memref<i32> -> %{{.+}} : memref<i32>) {
  omp.taskloop allocate(%testmemref : memref<i32> -> %testmemref : memref<i32>) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  %testi64 = "test.i64"() : () -> (i64)
  // CHECK: omp.taskloop grainsize(%{{[^:]+}}: i64) {
  omp.taskloop grainsize(%testi64: i64) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop num_tasks(%{{[^:]+}}: i64) {
  omp.taskloop num_tasks(%testi64: i64) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop nogroup {
  omp.taskloop nogroup {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      // CHECK: omp.yield
      omp.yield
    }
    omp.terminator
  }

  // CHECK: omp.taskloop {
  omp.taskloop {
    omp.simd {
      omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
        // CHECK: omp.yield
        omp.yield
      }
      omp.terminator
    } {omp.composite}
    omp.terminator
  } {omp.composite}

  // CHECK: return
  return
}

// CHECK: func.func @omp_requires_one
// CHECK-SAME: omp.requires = #omp<clause_requires reverse_offload>
func.func @omp_requires_one() -> ()
    attributes {omp.requires = #omp<clause_requires reverse_offload>} {
  return
}

// CHECK: func.func @omp_requires_multiple
// CHECK-SAME: omp.requires = #omp<clause_requires unified_address|dynamic_allocators>
func.func @omp_requires_multiple() -> ()
    attributes {omp.requires = #omp<clause_requires unified_address|dynamic_allocators>} {
  return
}

// CHECK-LABEL: @opaque_pointers_atomic_rwu
// CHECK-SAME: (%[[v:.*]]: !llvm.ptr, %[[x:.*]]: !llvm.ptr)
func.func @opaque_pointers_atomic_rwu(%v: !llvm.ptr, %x: !llvm.ptr) {
  // CHECK: omp.atomic.read %[[v]] = %[[x]] : !llvm.ptr, i32
  // CHECK: %[[VAL:.*]] = llvm.load %[[x]] : !llvm.ptr -> i32
  // CHECK: omp.atomic.write %[[v]] = %[[VAL]] : !llvm.ptr, i32
  // CHECK: omp.atomic.update %[[x]] : !llvm.ptr {
  // CHECK-NEXT: ^{{[[:alnum:]]+}}(%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   omp.yield(%[[XVAL]] : i32)
  // CHECK-NEXT: }
  omp.atomic.read %v = %x : !llvm.ptr, i32
  %val = llvm.load %x : !llvm.ptr -> i32
  omp.atomic.write %v = %val : !llvm.ptr, i32
  omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      omp.yield(%xval : i32)
  }
  return
}

// CHECK-LABEL: @opaque_pointers_reduction
// CHECK: atomic {
// CHECK-NEXT: ^{{[[:alnum:]]+}}(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr):
// CHECK-NOT: cleanup
omp.declare_reduction @opaque_pointers_reduction : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

// CHECK-LABEL: @alloc_reduction
// CHECK-SAME:  alloc {
// CHECK-NEXT:  ^bb0(%[[ARG0:.*]]: !llvm.ptr):
// ...
// CHECK:         omp.yield
// CHECK-NEXT:  } init {
// CHECK:       } combiner {
// CHECK:       }
omp.declare_reduction @alloc_reduction : !llvm.ptr
alloc {
^bb0(%arg: !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  omp.yield (%0 : !llvm.ptr)
}
init {
^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
  %cst = arith.constant 1.0 : f32
  llvm.store %cst, %alloc : f32, !llvm.ptr
  omp.yield (%alloc : !llvm.ptr)
}
combiner {
^bb1(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  %1 = llvm.load %arg1 : !llvm.ptr -> f32
  %2 = arith.addf %0, %1 : f32
  llvm.store %2, %arg0 : f32, !llvm.ptr
  omp.yield (%arg0 : !llvm.ptr)
}

// CHECK-LABEL: omp_targets_with_map_bounds
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr)
func.func @omp_targets_with_map_bounds(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> () {
  // CHECK: %[[C_00:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[C_01:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[C_02:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[C_03:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[BOUNDS0:.*]] = omp.map.bounds   lower_bound(%[[C_01]] : i64) upper_bound(%[[C_00]] : i64) stride(%[[C_02]] : i64) start_idx(%[[C_03]] : i64)
  // CHECK: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS0]]) -> !llvm.ptr {name = ""}
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = omp.map.bounds   lower_bound(%1 : i64) upper_bound(%0 : i64) stride(%2 : i64) start_idx(%3 : i64)

    %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%4) -> !llvm.ptr {name = ""}
  // CHECK: %[[C_10:.*]] = llvm.mlir.constant(9 : index) : i64
  // CHECK: %[[C_11:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[C_12:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[C_13:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[BOUNDS1:.*]] = omp.map.bounds   lower_bound(%[[C_11]] : i64) upper_bound(%[[C_10]] : i64) stride(%[[C_12]] : i64) start_idx(%[[C_13]] : i64)
  // CHECK: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ARG1]] : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByCopy) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
    %6 = llvm.mlir.constant(9 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(2 : index) : i64
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = omp.map.bounds   lower_bound(%7 : i64) upper_bound(%6 : i64) stride(%8 : i64) start_idx(%9 : i64)
    %mapv2 = omp.map.info var_ptr(%arg1 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(ByCopy) bounds(%10) -> !llvm.ptr {name = ""}

    // CHECK: omp.target map_entries(%[[MAP0]] -> {{.*}}, %[[MAP1]] -> {{.*}} : !llvm.ptr, !llvm.ptr)
    omp.target map_entries(%mapv1 -> %arg2, %mapv2 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }

    // CHECK: omp.target_data map_entries(%[[MAP0]], %[[MAP1]] : !llvm.ptr, !llvm.ptr)
    omp.target_data map_entries(%mapv1, %mapv2 : !llvm.ptr, !llvm.ptr){}

    // CHECK: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(VLAType) bounds(%[[BOUNDS0]]) -> !llvm.ptr {name = ""}
    // CHECK: omp.target_enter_data map_entries(%[[MAP2]] : !llvm.ptr)
    %mapv3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(VLAType) bounds(%4) -> !llvm.ptr {name = ""}
    omp.target_enter_data map_entries(%mapv3 : !llvm.ptr){}

    // CHECK: %[[MAP3:.*]] = omp.map.info var_ptr(%[[ARG1]] : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(This) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
    // CHECK: omp.target_exit_data map_entries(%[[MAP3]] : !llvm.ptr)
    %mapv4 = omp.map.info var_ptr(%arg1 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(exit_release_or_enter_alloc) capture(This) bounds(%10) -> !llvm.ptr {name = ""}
    omp.target_exit_data map_entries(%mapv4 : !llvm.ptr){}

    return
}

// CHECK-LABEL: omp_target_update_data
func.func @omp_target_update_data (%if_cond : i1, %device : si32, %map1: memref<?xi32>, %map2: memref<?xi32>) -> () {
    %mapv_from = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32> {name = ""}

    %mapv_to = omp.map.info var_ptr(%map2 : memref<?xi32>, tensor<?xi32>) map_clauses(present, to) capture(ByRef) -> memref<?xi32> {name = ""}

    // CHECK: omp.target_update device(%[[VAL_1:.*]] : si32) if(%[[VAL_0:.*]]) map_entries(%{{.*}}, %{{.*}} : memref<?xi32>, memref<?xi32>) nowait
    omp.target_update if(%if_cond) device(%device : si32) nowait map_entries(%mapv_from , %mapv_to : memref<?xi32>, memref<?xi32>)
    return
}

// CHECK-LABEL: omp_targets_is_allocatable
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr)
func.func @omp_targets_is_allocatable(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> () {
  // CHECK: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  // CHECK: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ARG1]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) members(%[[MAP0]] : [0] : !llvm.ptr) -> !llvm.ptr {name = ""}
  %mapv2 = omp.map.info var_ptr(%arg1 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>)   map_clauses(tofrom) capture(ByRef) members(%mapv1 : [0] : !llvm.ptr) -> !llvm.ptr {name = ""}
  // CHECK: omp.target map_entries(%[[MAP0]] -> {{.*}}, %[[MAP1]] -> {{.*}} : !llvm.ptr, !llvm.ptr)
  omp.target map_entries(%mapv1 -> %arg2, %mapv2 -> %arg3 : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func @omp_target_enter_update_exit_data_depend
// CHECK-SAME:([[ARG0:%.*]]: memref<?xi32>, [[ARG1:%.*]]: memref<?xi32>, [[ARG2:%.*]]: memref<?xi32>) {
func.func @omp_target_enter_update_exit_data_depend(%a: memref<?xi32>, %b: memref<?xi32>, %c: memref<?xi32>) {
// CHECK-NEXT: [[MAP0:%.*]] = omp.map.info
// CHECK-NEXT: [[MAP1:%.*]] = omp.map.info
// CHECK-NEXT: [[MAP2:%.*]] = omp.map.info
  %map_a = omp.map.info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32>
  %map_b = omp.map.info var_ptr(%b: memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  %map_c = omp.map.info var_ptr(%c: memref<?xi32>, tensor<?xi32>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32>

  // Do some work on the host that writes to 'a'
  omp.task depend(taskdependout -> %a : memref<?xi32>) {
    "test.foo"(%a) : (memref<?xi32>) -> ()
    omp.terminator
  }

  // Then map that over to the target
  // CHECK: omp.target_enter_data depend(taskdependin -> [[ARG0]] : memref<?xi32>) map_entries([[MAP0]], [[MAP2]] : memref<?xi32>, memref<?xi32>) nowait
  omp.target_enter_data depend(taskdependin ->  %a: memref<?xi32>) nowait map_entries(%map_a, %map_c: memref<?xi32>, memref<?xi32>)

  // Compute 'b' on the target and copy it back
  // CHECK: omp.target map_entries([[MAP1]] -> {{%.*}} : memref<?xi32>) {
  omp.target map_entries(%map_b -> %arg0 : memref<?xi32>) {
    "test.foo"(%arg0) : (memref<?xi32>) -> ()
    omp.terminator
  }

  // Update 'a' on the host using 'b'
  omp.task depend(taskdependout -> %a: memref<?xi32>){
    "test.bar"(%a, %b) : (memref<?xi32>, memref<?xi32>) -> ()
  }

  // Copy the updated 'a' onto the target
  // CHECK: omp.target_update depend(taskdependin -> [[ARG0]] : memref<?xi32>) map_entries([[MAP0]] : memref<?xi32>) nowait
  omp.target_update depend(taskdependin -> %a : memref<?xi32>) nowait map_entries(%map_a :  memref<?xi32>)

  // Compute 'c' on the target and copy it back
  %map_c_from = omp.map.info var_ptr(%c: memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  omp.target depend(taskdependout -> %c : memref<?xi32>) map_entries(%map_a -> %arg0, %map_c_from -> %arg1 : memref<?xi32>, memref<?xi32>) {
    "test.foobar"() : ()->()
    omp.terminator
  }
  // CHECK: omp.target_exit_data depend(taskdependin -> [[ARG2]] : memref<?xi32>) map_entries([[MAP2]] : memref<?xi32>)
  omp.target_exit_data depend(taskdependin -> %c : memref<?xi32>) map_entries(%map_c : memref<?xi32>)

  return
}

// CHECK-LABEL: omp_map_with_members
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr, %[[ARG3:.*]]: !llvm.ptr, %[[ARG4:.*]]: !llvm.ptr, %[[ARG5:.*]]: !llvm.ptr)
func.func @omp_map_with_members(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) -> () {
  // CHECK: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}
  %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ARG1]] : !llvm.ptr, f32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}
  %mapv2 = omp.map.info var_ptr(%arg1 : !llvm.ptr, f32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ARG2]] : !llvm.ptr, !llvm.struct<(i32, f32)>) map_clauses(to) capture(ByRef) members(%[[MAP0]], %[[MAP1]] : [0], [1] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "", partial_map = true}
  %mapv3 = omp.map.info var_ptr(%arg2 : !llvm.ptr, !llvm.struct<(i32, f32)>)   map_clauses(to) capture(ByRef) members(%mapv1, %mapv2 : [0], [1] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "", partial_map = true}

  // CHECK: omp.target_enter_data map_entries(%[[MAP0]], %[[MAP1]], %[[MAP2]] : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  omp.target_enter_data map_entries(%mapv1, %mapv2, %mapv3 : !llvm.ptr, !llvm.ptr, !llvm.ptr){}

  // CHECK: %[[MAP3:.*]] = omp.map.info var_ptr(%[[ARG3]] : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  %mapv4 = omp.map.info var_ptr(%arg3 : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: %[[MAP4:.*]] = omp.map.info var_ptr(%[[ARG4]] : !llvm.ptr, f32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}
  %mapv5 = omp.map.info var_ptr(%arg4 : !llvm.ptr, f32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: %[[MAP5:.*]] = omp.map.info var_ptr(%[[ARG5]] : !llvm.ptr, !llvm.struct<(i32, struct<(i32, f32)>)>) map_clauses(from) capture(ByRef) members(%[[MAP3]], %[[MAP4]] : [1,0], [1,1] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "", partial_map = true}
  %mapv6 = omp.map.info var_ptr(%arg5 : !llvm.ptr, !llvm.struct<(i32, struct<(i32, f32)>)>) map_clauses(from) capture(ByRef) members(%mapv4, %mapv5 : [1,0], [1,1] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "", partial_map = true}

  // CHECK: omp.target_exit_data map_entries(%[[MAP3]], %[[MAP4]], %[[MAP5]] : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  omp.target_exit_data map_entries(%mapv4, %mapv5, %mapv6 : !llvm.ptr, !llvm.ptr, !llvm.ptr){}

  return
}

// CHECK-LABEL: parallel_op_privatizers
// CHECK-SAME: (%[[ARG0:[^[:space:]]+]]: !llvm.ptr, %[[ARG1:[^[:space:]]+]]: !llvm.ptr)
func.func @parallel_op_privatizers(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK: omp.parallel private(
  // CHECK-SAME: @x.privatizer %[[ARG0]] -> %[[ARG0_PRIV:[^[:space:]]+]],
  // CHECK-SAME: @y.privatizer %[[ARG1]] -> %[[ARG1_PRIV:[^[:space:]]+]] : !llvm.ptr, !llvm.ptr)
  omp.parallel private(@x.privatizer %arg0 -> %arg2, @y.privatizer %arg1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
    // CHECK: llvm.load %[[ARG0_PRIV]]
    %0 = llvm.load %arg2 : !llvm.ptr -> i32
    // CHECK: llvm.load %[[ARG1_PRIV]]
    %1 = llvm.load %arg3 : !llvm.ptr -> i32
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp.private {type = private} @a.privatizer : !llvm.ptr alloc {
omp.private {type = private} @a.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%{{.*}}: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

// CHECK-LABEL: omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%{{.*}}: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} dealloc {
// CHECK: ^bb0(%{{.*}}: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield
}

// CHECK-LABEL: omp.private {type = firstprivate} @y.privatizer : !llvm.ptr alloc {
omp.private {type = firstprivate} @y.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%{{.*}}: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
// CHECK: } copy {
} copy {
// CHECK: ^bb0(%{{.*}}: {{.*}}, %{{.*}}: {{.*}}):
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} dealloc {
// CHECK: ^bb0(%{{.*}}: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield
}

// CHECK-LABEL: parallel_op_reduction_and_private
func.func @parallel_op_reduction_and_private(%priv_var: !llvm.ptr, %priv_var2: !llvm.ptr, %reduc_var: !llvm.ptr, %reduc_var2: !llvm.ptr) {
  // CHECK: omp.parallel
  // CHECK-SAME: private(
  // CHECK-SAME: @x.privatizer %[[PRIV_VAR:[^[:space:]]+]] -> %[[PRIV_ARG:[^[:space:]]+]],
  // CHECK-SAME: @y.privatizer %[[PRIV_VAR2:[^[:space:]]+]] -> %[[PRIV_ARG2:[^[:space:]]+]] : !llvm.ptr, !llvm.ptr)
  //
  // CHECK-SAME: reduction(
  // CHECK-SAME: @add_f32 %[[REDUC_VAR:[^[:space:]]+]] -> %[[REDUC_ARG:[^[:space:]]+]],
  // CHECK-SAME: @add_f32 %[[REDUC_VAR2:[^[:space:]]+]] -> %[[REDUC_ARG2:[^[:space:]]+]] : !llvm.ptr, !llvm.ptr)
  omp.parallel private(@x.privatizer %priv_var -> %priv_arg, @y.privatizer %priv_var2 -> %priv_arg2 : !llvm.ptr, !llvm.ptr)
               reduction(@add_f32 %reduc_var -> %reduc_arg, @add_f32 %reduc_var2 -> %reduc_arg2 : !llvm.ptr, !llvm.ptr) {
    // CHECK: llvm.load %[[PRIV_ARG]]
    %0 = llvm.load %priv_arg : !llvm.ptr -> f32
    // CHECK: llvm.load %[[PRIV_ARG2]]
    %1 = llvm.load %priv_arg2 : !llvm.ptr -> f32
    // CHECK: llvm.load %[[REDUC_ARG]]
    %2 = llvm.load %reduc_arg : !llvm.ptr -> f32
    // CHECK: llvm.load %[[REDUC_ARG2]]
    %3 = llvm.load %reduc_arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  return
}

// CHECK-LABEL: omp_target_private
func.func @omp_target_private(%map1: memref<?xi32>, %map2: memref<?xi32>, %priv_var: !llvm.ptr) -> () {
  %mapv1 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(tofrom) capture(ByRef) -> memref<?xi32> {name = ""}
  %mapv2 = omp.map.info var_ptr(%map2 : memref<?xi32>, tensor<?xi32>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}

  // CHECK: omp.target
  // CHECK-SAME: private(
  // CHECK-SAME:   @x.privatizer %{{[^[:space:]]+}} -> %[[PRIV_ARG:[^[:space:]]+]]
  // CHECK-SAME:   : !llvm.ptr
  // CHECK-SAME: )
  omp.target private(@x.privatizer %priv_var -> %priv_arg : !llvm.ptr) {
    omp.terminator
  }

  // CHECK: omp.target

  // CHECK-SAME: map_entries(
  // CHECK-SAME:   %{{[^[:space:]]+}} -> %[[MAP1_ARG:[^[:space:]]+]],
  // CHECK-SAME:   %{{[^[:space:]]+}} -> %[[MAP2_ARG:[^[:space:]]+]]
  // CHECK-SAME:   : memref<?xi32>, memref<?xi32>
  // CHECK-SAME: )

  // CHECK-SAME: private(
  // CHECK-SAME:   @x.privatizer %{{[^[:space:]]+}} -> %[[PRIV_ARG:[^[:space:]]+]]
  // CHECK-SAME:   : !llvm.ptr
  // CHECK-SAME: )
  omp.target map_entries(%mapv1 -> %arg0, %mapv2 -> %arg1 : memref<?xi32>, memref<?xi32>) private(@x.privatizer %priv_var -> %priv_arg : !llvm.ptr) {
    omp.terminator
  }

  return
}
