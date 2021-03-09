// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_stand_alone_directives()
llvm.func @test_stand_alone_directives() {
  // CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK-NEXT:  call void @__kmpc_barrier(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD]])
  omp.barrier

  // CHECK: [[OMP_THREAD1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK-NEXT:  [[RET_VAL:%.*]] = call i32 @__kmpc_omp_taskwait(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD1]])
  omp.taskwait

  // CHECK: [[OMP_THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK-NEXT:  [[RET_VAL:%.*]] = call i32 @__kmpc_omp_taskyield(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD2]], i32 0)
  omp.taskyield

  // CHECK-NEXT:    ret void
  llvm.return
}

// CHECK-LABEL: define void @test_flush_construct(i32 %0)
llvm.func @test_flush_construct(%arg0: i32) {
  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush

  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush (%arg0 : i32)

  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush (%arg0, %arg0 : i32, i32)

  %0 = llvm.mlir.constant(1 : i64) : i64
  //  CHECK: alloca {{.*}} align 4
  %1 = llvm.alloca %0 x i32 {in_type = i32, name = "a"} : (i64) -> !llvm.ptr<i32>
  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush
  //  CHECK: load i32, i32*
  %2 = llvm.load %1 : !llvm.ptr<i32>

  // CHECK-NEXT:    ret void
  llvm.return
}

// CHECK-LABEL: define void @test_omp_parallel_1()
llvm.func @test_omp_parallel_1() -> () {
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_1:.*]] to {{.*}}
  omp.parallel {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_1]]
  // CHECK: call void @__kmpc_barrier

llvm.func @body(i64)

// CHECK-LABEL: define void @test_omp_parallel_2()
llvm.func @test_omp_parallel_2() -> () {
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_2:.*]] to {{.*}}
  omp.parallel {
    ^bb0:
      %0 = llvm.mlir.constant(1 : index) : i64
      %1 = llvm.mlir.constant(42 : index) : i64
      llvm.call @body(%0) : (i64) -> ()
      llvm.call @body(%1) : (i64) -> ()
      llvm.br ^bb1

    ^bb1:
      %2 = llvm.add %0, %1 : i64
      llvm.call @body(%2) : (i64) -> ()
      omp.terminator
  }
  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_2]]
  // CHECK-LABEL: omp.par.region:
  // CHECK: br label %omp.par.region1
  // CHECK-LABEL: omp.par.region1:
  // CHECK: call void @body(i64 1)
  // CHECK: call void @body(i64 42)
  // CHECK: br label %omp.par.region2
  // CHECK-LABEL: omp.par.region2:
  // CHECK: call void @body(i64 43)
  // CHECK: br label %omp.par.pre_finalize

// CHECK: define void @test_omp_parallel_num_threads_1(i32 %[[NUM_THREADS_VAR_1:.*]])
llvm.func @test_omp_parallel_num_threads_1(%arg0: i32) -> () {
  // CHECK: %[[GTN_NUM_THREADS_VAR_1:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_1:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_1]], i32 %[[GTN_NUM_THREADS_VAR_1]], i32 %[[NUM_THREADS_VAR_1]])
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_1:.*]] to {{.*}}
  omp.parallel num_threads(%arg0: i32) {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_NUM_THREADS_1]]
  // CHECK: call void @__kmpc_barrier

// CHECK: define void @test_omp_parallel_num_threads_2()
llvm.func @test_omp_parallel_num_threads_2() -> () {
  %0 = llvm.mlir.constant(4 : index) : i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_2:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_2:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_2]], i32 %[[GTN_NUM_THREADS_VAR_2]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_2:.*]] to {{.*}}
  omp.parallel num_threads(%0: i32) {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_NUM_THREADS_2]]
  // CHECK: call void @__kmpc_barrier

// CHECK: define void @test_omp_parallel_num_threads_3()
llvm.func @test_omp_parallel_num_threads_3() -> () {
  %0 = llvm.mlir.constant(4 : index) : i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_3_1:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_3_1:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_3_1]], i32 %[[GTN_NUM_THREADS_VAR_3_1]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_3_1:.*]] to {{.*}}
  omp.parallel num_threads(%0: i32) {
    omp.barrier
    omp.terminator
  }
  %1 = llvm.mlir.constant(8 : index) : i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_3_2:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_3_2:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_3_2]], i32 %[[GTN_NUM_THREADS_VAR_3_2]], i32 8)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_3_2:.*]] to {{.*}}
  omp.parallel num_threads(%1: i32) {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_NUM_THREADS_3_2]]
  // CHECK: call void @__kmpc_barrier

// CHECK: define internal void @[[OMP_OUTLINED_FN_NUM_THREADS_3_1]]
  // CHECK: call void @__kmpc_barrier

// CHECK: define void @test_omp_parallel_if_1(i32 %[[IF_VAR_1:.*]])
llvm.func @test_omp_parallel_if_1(%arg0: i32) -> () {

// CHECK: %[[IF_COND_VAR_1:.*]] = icmp slt i32 %[[IF_VAR_1]], 0
  %0 = llvm.mlir.constant(0 : index) : i32
  %1 = llvm.icmp "slt" %arg0, %0 : i32

// CHECK: %[[GTN_IF_1:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[SI_VAR_IF_1:.*]])
// CHECK: br i1 %[[IF_COND_VAR_1]], label %[[IF_COND_TRUE_BLOCK_1:.*]], label %[[IF_COND_FALSE_BLOCK_1:.*]]
// CHECK: [[IF_COND_TRUE_BLOCK_1]]:
// CHECK: br label %[[OUTLINED_CALL_IF_BLOCK_1:.*]]
// CHECK: [[OUTLINED_CALL_IF_BLOCK_1]]:
// CHECK: call void {{.*}} @__kmpc_fork_call(%struct.ident_t* @[[SI_VAR_IF_1]], {{.*}} @[[OMP_OUTLINED_FN_IF_1:.*]] to void
// CHECK: br label %[[OUTLINED_EXIT_IF_1:.*]]
// CHECK: [[OUTLINED_EXIT_IF_1]]:
// CHECK: br label %[[OUTLINED_EXIT_IF_2:.*]]
// CHECK: [[OUTLINED_EXIT_IF_2]]:
// CHECK: br label %[[RETURN_BLOCK_IF_1:.*]]
// CHECK: [[IF_COND_FALSE_BLOCK_1]]:
// CHECK: call void @__kmpc_serialized_parallel(%struct.ident_t* @[[SI_VAR_IF_1]], i32 %[[GTN_IF_1]])
// CHECK: call void @[[OMP_OUTLINED_FN_IF_1]]
// CHECK: call void @__kmpc_end_serialized_parallel(%struct.ident_t* @[[SI_VAR_IF_1]], i32 %[[GTN_IF_1]])
// CHECK: br label %[[RETURN_BLOCK_IF_1]]
  omp.parallel if(%1 : i1) {
    omp.barrier
    omp.terminator
  }

// CHECK: [[RETURN_BLOCK_IF_1]]:
// CHECK: ret void
  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_IF_1]]
  // CHECK: call void @__kmpc_barrier

// CHECK-LABEL: define void @test_omp_parallel_3()
llvm.func @test_omp_parallel_3() -> () {
  // CHECK: [[OMP_THREAD_3_1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK: call void @__kmpc_push_proc_bind(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD_3_1]], i32 2)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_3_1:.*]] to {{.*}}
  omp.parallel proc_bind(master) {
    omp.barrier
    omp.terminator
  }
  // CHECK: [[OMP_THREAD_3_2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK: call void @__kmpc_push_proc_bind(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD_3_2]], i32 3)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_3_2:.*]] to {{.*}}
  omp.parallel proc_bind(close) {
    omp.barrier
    omp.terminator
  }
  // CHECK: [[OMP_THREAD_3_3:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK: call void @__kmpc_push_proc_bind(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD_3_3]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_3_3:.*]] to {{.*}}
  omp.parallel proc_bind(spread) {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_3_3]]
// CHECK: define internal void @[[OMP_OUTLINED_FN_3_2]]
// CHECK: define internal void @[[OMP_OUTLINED_FN_3_1]]

// CHECK-LABEL: define void @test_omp_parallel_4()
llvm.func @test_omp_parallel_4() -> () {
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_4_1:.*]] to
// CHECK: define internal void @[[OMP_OUTLINED_FN_4_1]]
// CHECK: call void @__kmpc_barrier
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_4_1_1:.*]] to
// CHECK: call void @__kmpc_barrier
  omp.parallel {
    omp.barrier

// CHECK: define internal void @[[OMP_OUTLINED_FN_4_1_1]]
// CHECK: call void @__kmpc_barrier
    omp.parallel {
      omp.barrier
      omp.terminator
    }

    omp.barrier
    omp.terminator
  }
  llvm.return
}

llvm.func @test_omp_parallel_5() -> () {
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_5_1:.*]] to
// CHECK: define internal void @[[OMP_OUTLINED_FN_5_1]]
// CHECK: call void @__kmpc_barrier
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_5_1_1:.*]] to
// CHECK: call void @__kmpc_barrier
  omp.parallel {
    omp.barrier

// CHECK: define internal void @[[OMP_OUTLINED_FN_5_1_1]]
    omp.parallel {
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_5_1_1_1:.*]] to
// CHECK: define internal void @[[OMP_OUTLINED_FN_5_1_1_1]]
// CHECK: call void @__kmpc_barrier
      omp.parallel {
        omp.barrier
        omp.terminator
      }
      omp.terminator
    }

    omp.barrier
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @test_omp_master()
llvm.func @test_omp_master() -> () {
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @{{.*}} to
// CHECK: omp.par.region1:
  omp.parallel {
    omp.master {
// CHECK: [[OMP_THREAD_3_4:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
// CHECK: {{[0-9]+}} = call i32 @__kmpc_master(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD_3_4]])
// CHECK: omp.master.region
// CHECK: call void @__kmpc_end_master(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD_3_4]])
// CHECK: br label %omp_region.end
      omp.terminator
    }
    omp.terminator
  }
  omp.parallel {
    omp.parallel {
      omp.master {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK: %struct.ident_t = type
// CHECK: @[[$parallel_loc:.*]] = private unnamed_addr constant {{.*}} c";LLVMDialectModule;wsloop_simple;{{[0-9]+}};{{[0-9]+}};;\00"
// CHECK: @[[$parallel_loc_struct:.*]] = private unnamed_addr constant %struct.ident_t {{.*}} @[[$parallel_loc]], {{.*}}

// CHECK: @[[$wsloop_loc:.*]] = private unnamed_addr constant {{.*}} c";LLVMDialectModule;wsloop_simple;{{[0-9]+}};{{[0-9]+}};;\00"
// CHECK: @[[$wsloop_loc_struct:.*]] = private unnamed_addr constant %struct.ident_t {{.*}} @[[$wsloop_loc]], {{.*}}

// CHECK-LABEL: @wsloop_simple
llvm.func @wsloop_simple(%arg0: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  omp.parallel {
    "omp.wsloop"(%1, %0, %2) ( {
    ^bb0(%arg1: i64):
      // The form of the emitted IR is controlled by OpenMPIRBuilder and
      // tested there. Just check that the right functions are called.
      // CHECK: call i32 @__kmpc_global_thread_num
      // CHECK: call void @__kmpc_for_static_init_{{.*}}(%struct.ident_t* @[[$wsloop_loc_struct]],
      %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
      %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %3, %4 : !llvm.ptr<f32>
      omp.yield
      // CHECK: call void @__kmpc_for_static_fini(%struct.ident_t* @[[$wsloop_loc_struct]],
    }) {operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0, 0, 0]> : vector<9xi32>} : (i64, i64, i64) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: @wsloop_inclusive_1
llvm.func @wsloop_inclusive_1(%arg0: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  // CHECK: store i64 31, i64* %{{.*}}upperbound
  "omp.wsloop"(%1, %0, %2) ( {
  ^bb0(%arg1: i64):
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %3, %4 : !llvm.ptr<f32>
    omp.yield
  }) {operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0, 0, 0]> : vector<9xi32>} : (i64, i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: @wsloop_inclusive_2
llvm.func @wsloop_inclusive_2(%arg0: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  // CHECK: store i64 32, i64* %{{.*}}upperbound
  "omp.wsloop"(%1, %0, %2) ( {
  ^bb0(%arg1: i64):
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %3, %4 : !llvm.ptr<f32>
    omp.yield
  }) {inclusive, operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0, 0, 0]> : vector<9xi32>} : (i64, i64, i64) -> ()
  llvm.return
}
