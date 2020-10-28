// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

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
llvm.func @test_flush_construct(%arg0: !llvm.i32) {
  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush

  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush %arg0 : !llvm.i32

  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{[0-9]+}}
  omp.flush %arg0, %arg0 : !llvm.i32, !llvm.i32

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

llvm.func @body(!llvm.i64)

// CHECK-LABEL: define void @test_omp_parallel_2()
llvm.func @test_omp_parallel_2() -> () {
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_2:.*]] to {{.*}}
  omp.parallel {
    ^bb0:
      %0 = llvm.mlir.constant(1 : index) : !llvm.i64
      %1 = llvm.mlir.constant(42 : index) : !llvm.i64
      llvm.call @body(%0) : (!llvm.i64) -> ()
      llvm.call @body(%1) : (!llvm.i64) -> ()
      llvm.br ^bb1

    ^bb1:
      %2 = llvm.add %0, %1 : !llvm.i64
      llvm.call @body(%2) : (!llvm.i64) -> ()
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
llvm.func @test_omp_parallel_num_threads_1(%arg0: !llvm.i32) -> () {
  // CHECK: %[[GTN_NUM_THREADS_VAR_1:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_1:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_1]], i32 %[[GTN_NUM_THREADS_VAR_1]], i32 %[[NUM_THREADS_VAR_1]])
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_1:.*]] to {{.*}}
  omp.parallel num_threads(%arg0: !llvm.i32) {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_NUM_THREADS_1]]
  // CHECK: call void @__kmpc_barrier

// CHECK: define void @test_omp_parallel_num_threads_2()
llvm.func @test_omp_parallel_num_threads_2() -> () {
  %0 = llvm.mlir.constant(4 : index) : !llvm.i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_2:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_2:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_2]], i32 %[[GTN_NUM_THREADS_VAR_2]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_2:.*]] to {{.*}}
  omp.parallel num_threads(%0: !llvm.i32) {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define internal void @[[OMP_OUTLINED_FN_NUM_THREADS_2]]
  // CHECK: call void @__kmpc_barrier

// CHECK: define void @test_omp_parallel_num_threads_3()
llvm.func @test_omp_parallel_num_threads_3() -> () {
  %0 = llvm.mlir.constant(4 : index) : !llvm.i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_3_1:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_3_1:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_3_1]], i32 %[[GTN_NUM_THREADS_VAR_3_1]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_3_1:.*]] to {{.*}}
  omp.parallel num_threads(%0: !llvm.i32) {
    omp.barrier
    omp.terminator
  }
  %1 = llvm.mlir.constant(8 : index) : !llvm.i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_3_2:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GTN_SI_VAR_3_2:.*]])
  // CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @[[GTN_SI_VAR_3_2]], i32 %[[GTN_NUM_THREADS_VAR_3_2]], i32 8)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_3_2:.*]] to {{.*}}
  omp.parallel num_threads(%1: !llvm.i32) {
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
llvm.func @test_omp_parallel_if_1(%arg0: !llvm.i32) -> () {

// CHECK: %[[IF_COND_VAR_1:.*]] = icmp slt i32 %[[IF_VAR_1]], 0
  %0 = llvm.mlir.constant(0 : index) : !llvm.i32
  %1 = llvm.icmp "slt" %arg0, %0 : !llvm.i32

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
  omp.parallel if(%1 : !llvm.i1) {
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
