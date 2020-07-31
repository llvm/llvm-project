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
