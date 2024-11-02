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

// Check that the allocas are emitted by the OpenMPIRBuilder at the top of the
// function, before the condition. Allocas are only emitted by the builder when
// the `if` clause is present. We match specific SSA value names since LLVM
// actually produces those names.
// CHECK: %tid.addr{{.*}} = alloca i32
// CHECK: %zero.addr{{.*}} = alloca i32

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

// -----

// CHECK-LABEL: @test_nested_alloca_ip
llvm.func @test_nested_alloca_ip(%arg0: i32) -> () {

  // Check that the allocas are emitted by the OpenMPIRBuilder at the top of
  // the function, before the condition. Allocas are only emitted by the
  // builder when the `if` clause is present. We match specific SSA value names
  // since LLVM actually produces those names and ensure they come before the
  // "icmp" that is the first operation we emit.
  // CHECK: %tid.addr{{.*}} = alloca i32
  // CHECK: %zero.addr{{.*}} = alloca i32
  // CHECK: icmp slt i32 %{{.*}}, 0
  %0 = llvm.mlir.constant(0 : index) : i32
  %1 = llvm.icmp "slt" %arg0, %0 : i32

  omp.parallel if(%1 : i1) {
    // The "parallel" operation will be outlined, check the the function is
    // produced. Inside that function, further allocas should be placed before
    // another "icmp".
    // CHECK: define
    // CHECK: %tid.addr{{.*}} = alloca i32
    // CHECK: %zero.addr{{.*}} = alloca i32
    // CHECK: icmp slt i32 %{{.*}}, 1
    %2 = llvm.mlir.constant(1 : index) : i32
    %3 = llvm.icmp "slt" %arg0, %2 : i32

    omp.parallel if(%3 : i1) {
      // One more nesting level.
      // CHECK: define
      // CHECK: %tid.addr{{.*}} = alloca i32
      // CHECK: %zero.addr{{.*}} = alloca i32
      // CHECK: icmp slt i32 %{{.*}}, 2

      %4 = llvm.mlir.constant(2 : index) : i32
      %5 = llvm.icmp "slt" %arg0, %4 : i32

      omp.parallel if(%5 : i1) {
        omp.barrier
        omp.terminator
      }

      omp.barrier
      omp.terminator
    }
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// -----

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
    "omp.wsloop"(%1, %0, %2) ({
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
    }) {operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0]> : vector<7xi32>} : (i64, i64, i64) -> ()
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL: @wsloop_inclusive_1
llvm.func @wsloop_inclusive_1(%arg0: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  // CHECK: store i64 31, i64* %{{.*}}upperbound
  "omp.wsloop"(%1, %0, %2) ({
  ^bb0(%arg1: i64):
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %3, %4 : !llvm.ptr<f32>
    omp.yield
  }) {operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0]> : vector<7xi32>} : (i64, i64, i64) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @wsloop_inclusive_2
llvm.func @wsloop_inclusive_2(%arg0: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  // CHECK: store i64 32, i64* %{{.*}}upperbound
  "omp.wsloop"(%1, %0, %2) ({
  ^bb0(%arg1: i64):
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %3, %4 : !llvm.ptr<f32>
    omp.yield
  }) {inclusive, operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0]> : vector<7xi32>} : (i64, i64, i64) -> ()
  llvm.return
}

// -----

llvm.func @body(i32)

// CHECK-LABEL: @test_omp_wsloop_static_defchunk
llvm.func @test_omp_wsloop_static_defchunk(%lb : i32, %ub : i32, %step : i32) -> () {
 omp.wsloop schedule(static)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
   // CHECK: call void @__kmpc_for_static_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 34, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32 1, i32 0)
   // CHECK: call void @__kmpc_for_static_fini
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i32)

// CHECK-LABEL: @test_omp_wsloop_static_1
llvm.func @test_omp_wsloop_static_1(%lb : i32, %ub : i32, %step : i32) -> () {
 %static_chunk_size = llvm.mlir.constant(1 : i32) : i32
 omp.wsloop schedule(static = %static_chunk_size : i32)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
   // CHECK: call void @__kmpc_for_static_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 33, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32 1, i32 1)
   // CHECK: call void @__kmpc_for_static_fini
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i32)

// CHECK-LABEL: @test_omp_wsloop_static_2
llvm.func @test_omp_wsloop_static_2(%lb : i32, %ub : i32, %step : i32) -> () {
 %static_chunk_size = llvm.mlir.constant(2 : i32) : i32
 omp.wsloop schedule(static = %static_chunk_size : i32)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
   // CHECK: call void @__kmpc_for_static_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 33, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32 1, i32 2)
   // CHECK: call void @__kmpc_for_static_fini
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(dynamic)
 for (%iv) : i64 = (%lb) to (%ub) step (%step)  {
  // CHECK: call void @__kmpc_dispatch_init_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK: br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic_chunk_const(%lb : i64, %ub : i64, %step : i64) -> () {
 %chunk_size_const = llvm.mlir.constant(2 : i16) : i16
 omp.wsloop schedule(dynamic = %chunk_size_const : i16)
 for (%iv) : i64 = (%lb) to (%ub) step (%step)  {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741859, i64 {{.*}}, i64 %{{.*}}, i64 {{.*}}, i64 2)
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK: br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i32)

llvm.func @test_omp_wsloop_dynamic_chunk_var(%lb : i32, %ub : i32, %step : i32) -> () {
 %1 = llvm.mlir.constant(1 : i64) : i64
 %chunk_size_alloca = llvm.alloca %1 x i16 {bindc_name = "chunk_size", in_type = i16, uniq_name = "_QFsub1Echunk_size"} : (i64) -> !llvm.ptr<i16>
 %chunk_size_var = llvm.load %chunk_size_alloca : !llvm.ptr<i16>
 omp.wsloop schedule(dynamic = %chunk_size_var : i16)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
  // CHECK: %[[CHUNK_SIZE:.*]] = sext i16 %{{.*}} to i32
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741859, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %[[CHUNK_SIZE]])
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_4u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK: br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i32)

llvm.func @test_omp_wsloop_dynamic_chunk_var2(%lb : i32, %ub : i32, %step : i32) -> () {
 %1 = llvm.mlir.constant(1 : i64) : i64
 %chunk_size_alloca = llvm.alloca %1 x i64 {bindc_name = "chunk_size", in_type = i64, uniq_name = "_QFsub1Echunk_size"} : (i64) -> !llvm.ptr<i64>
 %chunk_size_var = llvm.load %chunk_size_alloca : !llvm.ptr<i64>
 omp.wsloop schedule(dynamic = %chunk_size_var : i64)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
  // CHECK: %[[CHUNK_SIZE:.*]] = trunc i64 %{{.*}} to i32
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741859, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %[[CHUNK_SIZE]])
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_4u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK: br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i32)

llvm.func @test_omp_wsloop_dynamic_chunk_var3(%lb : i32, %ub : i32, %step : i32, %chunk_size : i32) -> () {
 omp.wsloop schedule(dynamic = %chunk_size : i32)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741859, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %{{.*}})
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_4u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK: br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_auto(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(auto)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_runtime(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop schedule(runtime)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    // CHECK: call void @__kmpc_dispatch_init_8u
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_guided(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop schedule(guided)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    // CHECK: call void @__kmpc_dispatch_init_8u
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic_nonmonotonic(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop schedule(dynamic, nonmonotonic)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741859
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic_monotonic(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop schedule(dynamic, monotonic)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 536870947
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_runtime_simd(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop schedule(runtime, simd)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741871
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_guided_simd(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop schedule(guided, simd)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741870
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

// CHECK-LABEL: @simdloop_simple
llvm.func @simdloop_simple(%lb : i64, %ub : i64, %step : i64, %arg0: !llvm.ptr<f32>) {
  "omp.simdloop" (%lb, %ub, %step) ({
    ^bb0(%iv: i64):
      %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
      // The form of the emitted IR is controlled by OpenMPIRBuilder and
      // tested there. Just check that the right metadata is added.
      // CHECK: llvm.access.group
      %4 = llvm.getelementptr %arg0[%iv] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %3, %4 : !llvm.ptr<f32>
      omp.yield
  }) {operand_segment_sizes = dense<[1,1,1]> : vector<3xi32>} :
    (i64, i64, i64) -> () 

  llvm.return
}
// CHECK: llvm.loop.parallel_accesses
// CHECK-NEXT: llvm.loop.vectorize.enable

// -----

// CHECK-LABEL: @simdloop_simple_multiple
llvm.func @simdloop_simple_multiple(%lb1 : i64, %ub1 : i64, %step1 : i64, %lb2 : i64, %ub2 : i64, %step2 : i64, %arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>) {
  omp.simdloop (%iv1, %iv2) : i64 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    // The form of the emitted IR is controlled by OpenMPIRBuilder and
    // tested there. Just check that the right metadata is added.
    // CHECK: llvm.access.group
    // CHECK-NEXT: llvm.access.group
    %4 = llvm.getelementptr %arg0[%iv1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %5 = llvm.getelementptr %arg1[%iv2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %3, %4 : !llvm.ptr<f32>
    llvm.store %3, %5 : !llvm.ptr<f32>
    omp.yield
  } 
  llvm.return
}
// CHECK: llvm.loop.parallel_accesses
// CHECK-NEXT: llvm.loop.vectorize.enable

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 66, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_static_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(static) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 66, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i32)

llvm.func @test_omp_wsloop_static_chunk_ordered(%lb : i32, %ub : i32, %step : i32) -> () {
 %static_chunk_size = llvm.mlir.constant(1 : i32) : i32
 omp.wsloop schedule(static = %static_chunk_size : i32) ordered(0)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 65, i32 1, i32 %{{.*}}, i32 1, i32 1)
  // CHECK: call void @__kmpc_dispatch_fini_4u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_4u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i32) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(dynamic) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 67, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_auto_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(auto) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 70, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_runtime_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(runtime) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 69, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_guided_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(guided) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 68, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic_nonmonotonic_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(dynamic, nonmonotonic) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 1073741891, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_dynamic_monotonic_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop schedule(dynamic, monotonic) ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 536870979, i64 1, i64 %{{.*}}, i64 1, i64 1)
  // CHECK: call void @__kmpc_dispatch_fini_8u
  // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
  // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
  // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
   llvm.call @body(%iv) : (i64) -> ()
   omp.yield
 }
 llvm.return
}

// -----

omp.critical.declare @mutex_none hint(none) // 0
omp.critical.declare @mutex_uncontended hint(uncontended) // 1
omp.critical.declare @mutex_contended hint(contended) // 2
omp.critical.declare @mutex_nonspeculative hint(nonspeculative) // 4
omp.critical.declare @mutex_nonspeculative_uncontended hint(nonspeculative, uncontended) // 5
omp.critical.declare @mutex_nonspeculative_contended hint(nonspeculative, contended) // 6
omp.critical.declare @mutex_speculative hint(speculative) // 8
omp.critical.declare @mutex_speculative_uncontended hint(speculative, uncontended) // 9
omp.critical.declare @mutex_speculative_contended hint(speculative, contended) // 10

// CHECK-LABEL: @omp_critical
llvm.func @omp_critical(%x : !llvm.ptr<i32>, %xval : i32) -> () {
  // CHECK: call void @__kmpc_critical({{.*}}critical_user_.var{{.*}})
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_none.var{{.*}}, i32 0)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_none) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_none.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_uncontended.var{{.*}}, i32 1)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_uncontended) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_uncontended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_contended.var{{.*}}, i32 2)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_contended) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_contended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_nonspeculative.var{{.*}}, i32 4)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_nonspeculative) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_nonspeculative.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_nonspeculative_uncontended.var{{.*}}, i32 5)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_nonspeculative_uncontended) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_nonspeculative_uncontended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_nonspeculative_contended.var{{.*}}, i32 6)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_nonspeculative_contended) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_nonspeculative_contended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_speculative.var{{.*}}, i32 8)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_speculative) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_speculative.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_speculative_uncontended.var{{.*}}, i32 9)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_speculative_uncontended) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_speculative_uncontended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_speculative_contended.var{{.*}}, i32 10)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_speculative_contended) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_speculative_contended.var{{.*}})
  llvm.return
}

// -----

// Check that the loop bounds are emitted in the correct location in case of
// collapse. This only checks the overall shape of the IR, detailed checking
// is done by the OpenMPIRBuilder.

// CHECK-LABEL: @collapse_wsloop
// CHECK: i32* noalias %[[TIDADDR:[0-9A-Za-z.]*]]
// CHECK: load i32, i32* %[[TIDADDR]]
// CHECK: store
// CHECK: load
// CHECK: %[[LB0:.*]] = load i32
// CHECK: %[[UB0:.*]] = load i32
// CHECK: %[[STEP0:.*]] = load i32
// CHECK: %[[LB1:.*]] = load i32
// CHECK: %[[UB1:.*]] = load i32
// CHECK: %[[STEP1:.*]] = load i32
// CHECK: %[[LB2:.*]] = load i32
// CHECK: %[[UB2:.*]] = load i32
// CHECK: %[[STEP2:.*]] = load i32
llvm.func @collapse_wsloop(
    %0: i32, %1: i32, %2: i32,
    %3: i32, %4: i32, %5: i32,
    %6: i32, %7: i32, %8: i32,
    %20: !llvm.ptr<i32>) {
  omp.parallel {
    // CHECK: icmp slt i32 %[[LB0]], 0
    // CHECK-COUNT-4: select
    // CHECK: %[[TRIPCOUNT0:.*]] = select
    // CHECK: br label %[[PREHEADER:.*]]
    //
    // CHECK: [[PREHEADER]]:
    // CHECK: icmp slt i32 %[[LB1]], 0
    // CHECK-COUNT-4: select
    // CHECK: %[[TRIPCOUNT1:.*]] = select
    // CHECK: icmp slt i32 %[[LB2]], 0
    // CHECK-COUNT-4: select
    // CHECK: %[[TRIPCOUNT2:.*]] = select
    // CHECK: %[[PROD:.*]] = mul nuw i32 %[[TRIPCOUNT0]], %[[TRIPCOUNT1]]
    // CHECK: %[[TOTAL:.*]] = mul nuw i32 %[[PROD]], %[[TRIPCOUNT2]]
    // CHECK: br label %[[COLLAPSED_PREHEADER:.*]]
    //
    // CHECK: [[COLLAPSED_PREHEADER]]:
    // CHECK: store i32 0, i32*
    // CHECK: %[[TOTAL_SUB_1:.*]] = sub i32 %[[TOTAL]], 1
    // CHECK: store i32 %[[TOTAL_SUB_1]], i32*
    // CHECK: call void @__kmpc_for_static_init_4u
    omp.wsloop collapse(3)
    for (%arg0, %arg1, %arg2) : i32 = (%0, %1, %2) to (%3, %4, %5) step (%6, %7, %8) {
      %31 = llvm.load %20 : !llvm.ptr<i32>
      %32 = llvm.add %31, %arg0 : i32
      %33 = llvm.add %32, %arg1 : i32
      %34 = llvm.add %33, %arg2 : i32
      llvm.store %34, %20 : !llvm.ptr<i32>
      omp.yield
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Check that the loop bounds are emitted in the correct location in case of
// collapse for dynamic schedule. This only checks the overall shape of the IR,
// detailed checking is done by the OpenMPIRBuilder.

// CHECK-LABEL: @collapse_wsloop_dynamic
// CHECK: i32* noalias %[[TIDADDR:[0-9A-Za-z.]*]]
// CHECK: load i32, i32* %[[TIDADDR]]
// CHECK: store
// CHECK: load
// CHECK: %[[LB0:.*]] = load i32
// CHECK: %[[UB0:.*]] = load i32
// CHECK: %[[STEP0:.*]] = load i32
// CHECK: %[[LB1:.*]] = load i32
// CHECK: %[[UB1:.*]] = load i32
// CHECK: %[[STEP1:.*]] = load i32
// CHECK: %[[LB2:.*]] = load i32
// CHECK: %[[UB2:.*]] = load i32
// CHECK: %[[STEP2:.*]] = load i32

llvm.func @collapse_wsloop_dynamic(
    %0: i32, %1: i32, %2: i32,
    %3: i32, %4: i32, %5: i32,
    %6: i32, %7: i32, %8: i32,
    %20: !llvm.ptr<i32>) {
  omp.parallel {
    // CHECK: icmp slt i32 %[[LB0]], 0
    // CHECK-COUNT-4: select
    // CHECK: %[[TRIPCOUNT0:.*]] = select
    // CHECK: br label %[[PREHEADER:.*]]
    //
    // CHECK: [[PREHEADER]]:
    // CHECK: icmp slt i32 %[[LB1]], 0
    // CHECK-COUNT-4: select
    // CHECK: %[[TRIPCOUNT1:.*]] = select
    // CHECK: icmp slt i32 %[[LB2]], 0
    // CHECK-COUNT-4: select
    // CHECK: %[[TRIPCOUNT2:.*]] = select
    // CHECK: %[[PROD:.*]] = mul nuw i32 %[[TRIPCOUNT0]], %[[TRIPCOUNT1]]
    // CHECK: %[[TOTAL:.*]] = mul nuw i32 %[[PROD]], %[[TRIPCOUNT2]]
    // CHECK: br label %[[COLLAPSED_PREHEADER:.*]]
    //
    // CHECK: [[COLLAPSED_PREHEADER]]:
    // CHECK: store i32 1, i32*
    // CHECK: store i32 %[[TOTAL]], i32*
    // CHECK: call void @__kmpc_dispatch_init_4u
    omp.wsloop collapse(3) schedule(dynamic)
    for (%arg0, %arg1, %arg2) : i32 = (%0, %1, %2) to (%3, %4, %5) step (%6, %7, %8) {
      %31 = llvm.load %20 : !llvm.ptr<i32>
      %32 = llvm.add %31, %arg0 : i32
      %33 = llvm.add %32, %arg1 : i32
      %34 = llvm.add %33, %arg2 : i32
      llvm.store %34, %20 : !llvm.ptr<i32>
      omp.yield
    }
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL: @omp_ordered
llvm.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64,
    %arg4: i64, %arg5: i64, %arg6: i64) -> () {
  // CHECK: [[ADDR9:%.*]] = alloca [2 x i64], align 8
  // CHECK: [[ADDR7:%.*]] = alloca [2 x i64], align 8
  // CHECK: [[ADDR5:%.*]] = alloca [2 x i64], align 8
  // CHECK: [[ADDR3:%.*]] = alloca [1 x i64], align 8
  // CHECK: [[ADDR:%.*]] = alloca [1 x i64], align 8

  // CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
  // CHECK-NEXT:  call void @__kmpc_ordered(%struct.ident_t* @[[GLOB1]], i32 [[OMP_THREAD]])
  omp.ordered_region {
    omp.terminator
  // CHECK: call void @__kmpc_end_ordered(%struct.ident_t* @[[GLOB1]], i32 [[OMP_THREAD]])
  }

  omp.wsloop ordered(0)
  for (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) {
    // CHECK:  call void @__kmpc_ordered(%struct.ident_t* @[[GLOB3:[0-9]+]], i32 [[OMP_THREAD2:%.*]])
    omp.ordered_region  {
      omp.terminator
    // CHECK: call void @__kmpc_end_ordered(%struct.ident_t* @[[GLOB3]], i32 [[OMP_THREAD2]])
    }
    omp.yield
  }

  omp.wsloop ordered(1)
  for (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) {
    // CHECK: [[TMP:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR]], i64 0, i64 0
    // CHECK: store i64 [[ARG0:%.*]], i64* [[TMP]], align 8
    // CHECK: [[TMP2:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR]], i64 0, i64 0
    // CHECK: [[OMP_THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB3:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_wait(%struct.ident_t* @[[GLOB3]], i32 [[OMP_THREAD2]], i64* [[TMP2]])
    omp.ordered depend_type(dependsink) depend_vec(%arg3 : i64) {num_loops_val = 1 : i64}

    // CHECK: [[TMP3:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR3]], i64 0, i64 0
    // CHECK: store i64 [[ARG0]], i64* [[TMP3]], align 8
    // CHECK: [[TMP4:%.*]] = getelementptr inbounds [1 x i64], [1 x i64]* [[ADDR3]], i64 0, i64 0
    // CHECK: [[OMP_THREAD4:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB5:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_post(%struct.ident_t* @[[GLOB5]], i32 [[OMP_THREAD4]], i64* [[TMP4]])
    omp.ordered depend_type(dependsource) depend_vec(%arg3 : i64) {num_loops_val = 1 : i64}

    omp.yield
  }

  omp.wsloop ordered(2)
  for (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) {
    // CHECK: [[TMP5:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR5]], i64 0, i64 0
    // CHECK: store i64 [[ARG0]], i64* [[TMP5]], align 8
    // CHECK: [[TMP6:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR5]], i64 0, i64 1
    // CHECK: store i64 [[ARG1:%.*]], i64* [[TMP6]], align 8
    // CHECK: [[TMP7:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR5]], i64 0, i64 0
    // CHECK: [[OMP_THREAD6:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB7:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_wait(%struct.ident_t* @[[GLOB7]], i32 [[OMP_THREAD6]], i64* [[TMP7]])
    // CHECK: [[TMP8:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR7]], i64 0, i64 0
    // CHECK: store i64 [[ARG2:%.*]], i64* [[TMP8]], align 8
    // CHECK: [[TMP9:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR7]], i64 0, i64 1
    // CHECK: store i64 [[ARG3:%.*]], i64* [[TMP9]], align 8
    // CHECK: [[TMP10:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR7]], i64 0, i64 0
    // CHECK: [[OMP_THREAD8:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB7]])
    // CHECK: call void @__kmpc_doacross_wait(%struct.ident_t* @[[GLOB7]], i32 [[OMP_THREAD8]], i64* [[TMP10]])
    omp.ordered depend_type(dependsink) depend_vec(%arg3, %arg4, %arg5, %arg6 : i64, i64, i64, i64) {num_loops_val = 2 : i64}

    // CHECK: [[TMP11:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR9]], i64 0, i64 0
    // CHECK: store i64 [[ARG0]], i64* [[TMP11]], align 8
    // CHECK: [[TMP12:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR9]], i64 0, i64 1
    // CHECK: store i64 [[ARG1]], i64* [[TMP12]], align 8
    // CHECK: [[TMP13:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[ADDR9]], i64 0, i64 0
    // CHECK: [[OMP_THREAD10:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB9:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_post(%struct.ident_t* @[[GLOB9]], i32 [[OMP_THREAD10]], i64* [[TMP13]])
    omp.ordered depend_type(dependsource) depend_vec(%arg3, %arg4 : i64, i64) {num_loops_val = 2 : i64}

    omp.yield
  }

  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_read
// CHECK-SAME: (i32* %[[ARG0:.*]], i32* %[[ARG1:.*]])
llvm.func @omp_atomic_read(%arg0 : !llvm.ptr<i32>, %arg1 : !llvm.ptr<i32>) -> () {

  // CHECK: %[[X1:.*]] = load atomic i32, i32* %[[ARG0]] monotonic, align 4
  // CHECK: store i32 %[[X1]], i32* %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 : !llvm.ptr<i32>

  // CHECK: %[[X2:.*]] = load atomic i32, i32* %[[ARG0]] seq_cst, align 4
  // CHECK: call void @__kmpc_flush(%{{.*}})
  // CHECK: store i32 %[[X2]], i32* %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 memory_order(seq_cst) : !llvm.ptr<i32>

  // CHECK: %[[X3:.*]] = load atomic i32, i32* %[[ARG0]] acquire, align 4
  // CHECK: call void @__kmpc_flush(%{{.*}})
  // CHECK: store i32 %[[X3]], i32* %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 memory_order(acquire) : !llvm.ptr<i32>

  // CHECK: %[[X4:.*]] = load atomic i32, i32* %[[ARG0]] monotonic, align 4
  // CHECK: store i32 %[[X4]], i32* %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 memory_order(relaxed) : !llvm.ptr<i32>
  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_write
// CHECK-SAME: (i32* %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_write(%x: !llvm.ptr<i32>, %expr: i32) -> () {
  // CHECK: store atomic i32 %[[expr]], i32* %[[x]] monotonic, align 4
  omp.atomic.write %x = %expr : !llvm.ptr<i32>, i32
  // CHECK: store atomic i32 %[[expr]], i32* %[[x]] seq_cst, align 4
  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{.*}})
  omp.atomic.write %x = %expr memory_order(seq_cst) : !llvm.ptr<i32>, i32
  // CHECK: store atomic i32 %[[expr]], i32* %[[x]] release, align 4
  // CHECK: call void @__kmpc_flush(%struct.ident_t* @{{.*}})
  omp.atomic.write %x = %expr memory_order(release) : !llvm.ptr<i32>, i32
  // CHECK: store atomic i32 %[[expr]], i32* %[[x]] monotonic, align 4
  omp.atomic.write %x = %expr memory_order(relaxed) : !llvm.ptr<i32>, i32
  llvm.return
}

// -----

// Checking simple atomicrmw and cmpxchg based translation. This also checks for
// ambigous alloca insert point by putting llvm.mul as the first update operation.
// CHECK-LABEL: @omp_atomic_update
// CHECK-SAME: (i32* %[[x:.*]], i32 %[[expr:.*]], i1* %[[xbool:.*]], i1 %[[exprbool:.*]])
llvm.func @omp_atomic_update(%x:!llvm.ptr<i32>, %expr: i32, %xbool: !llvm.ptr<i1>, %exprbool: i1) {
  // CHECK: %[[t1:.*]] = mul i32 %[[x_old:.*]], %[[expr]]
  // CHECK: store i32 %[[t1]], i32* %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, i32* %[[x_new]]
  // CHECK: cmpxchg i32* %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.mul %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  // CHECK: atomicrmw add i32* %[[x]], i32 %[[expr]] monotonic
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking an order-dependent operation when the order is `expr binop x`
// CHECK-LABEL: @omp_atomic_update_ordering
// CHECK-SAME: (i32* %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_update_ordering(%x:!llvm.ptr<i32>, %expr: i32) {
  // CHECK: %[[t1:.*]] = shl i32 %[[expr]], %[[x_old:[^ ,]*]]
  // CHECK: store i32 %[[t1]], i32* %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, i32* %[[x_new]]
  // CHECK: cmpxchg i32* %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.shl %expr, %xval : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking an order-dependent operation when the order is `x binop expr`
// CHECK-LABEL: @omp_atomic_update_ordering
// CHECK-SAME: (i32* %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_update_ordering(%x:!llvm.ptr<i32>, %expr: i32) {
  // CHECK: %[[t1:.*]] = shl i32 %[[x_old:.*]], %[[expr]]
  // CHECK: store i32 %[[t1]], i32* %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, i32* %[[x_new]]
  // CHECK: cmpxchg i32* %[[x]], i32 %[[x_old]], i32 %[[t2]] monotonic
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.shl %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking intrinsic translation.
// CHECK-LABEL: @omp_atomic_update_intrinsic
// CHECK-SAME: (i32* %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_update_intrinsic(%x:!llvm.ptr<i32>, %expr: i32) {
  // CHECK: %[[t1:.*]] = call i32 @llvm.smax.i32(i32 %[[x_old:.*]], i32 %[[expr]])
  // CHECK: store i32 %[[t1]], i32* %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, i32* %[[x_new]]
  // CHECK: cmpxchg i32* %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = "llvm.intr.smax"(%xval, %expr) : (i32, i32) -> i32
    omp.yield(%newval : i32)
  }
  // CHECK: %[[t1:.*]] = call i32 @llvm.umax.i32(i32 %[[x_old:.*]], i32 %[[expr]])
  // CHECK: store i32 %[[t1]], i32* %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, i32* %[[x_new]]
  // CHECK: cmpxchg i32* %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = "llvm.intr.umax"(%xval, %expr) : (i32, i32) -> i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_capture_prefix_update
// CHECK-SAME: (i32* %[[x:.*]], i32* %[[v:.*]], i32 %[[expr:.*]], float* %[[xf:.*]], float* %[[vf:.*]], float %[[exprf:.*]])
llvm.func @omp_atomic_capture_prefix_update(
  %x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>, %expr: i32,
  %xf: !llvm.ptr<f32>, %vf: !llvm.ptr<f32>, %exprf: f32) -> () {
  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = add i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[res:.*]] = atomicrmw sub i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = sub i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.sub %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[res:.*]] = atomicrmw and i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = and i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.and %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[res:.*]] = atomicrmw or i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = or i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.or %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[res:.*]] = atomicrmw xor i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = xor i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.xor %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = mul i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.mul %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = sdiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.sdiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = udiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.udiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = shl i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.shl %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = lshr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.lshr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = ashr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.ashr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr<i32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[newval:.*]] = fadd float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], float* %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK: %[[xf_bitcast:.*]] = bitcast float* %[[xf]] to i32*
  // CHECK: %{{.*}} = cmpxchg i32* %[[xf_bitcast]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[newval]], float* %[[vf]]
  omp.atomic.capture {
    omp.atomic.update %xf : !llvm.ptr<f32> {
    ^bb0(%xval: f32):
      %newval = llvm.fadd %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
    omp.atomic.read %vf = %xf : !llvm.ptr<f32>
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[newval:.*]] = fsub float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], float* %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK: %[[xf_bitcast:.*]] = bitcast float* %[[xf]] to i32*
  // CHECK: %{{.*}} = cmpxchg i32* %[[xf_bitcast]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[newval]], float* %[[vf]]
  omp.atomic.capture {
    omp.atomic.update %xf : !llvm.ptr<f32> {
    ^bb0(%xval: f32):
      %newval = llvm.fsub %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
    omp.atomic.read %vf = %xf : !llvm.ptr<f32>
  }

  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_capture_postfix_update
// CHECK-SAME: (i32* %[[x:.*]], i32* %[[v:.*]], i32 %[[expr:.*]], float* %[[xf:.*]], float* %[[vf:.*]], float %[[exprf:.*]])
llvm.func @omp_atomic_capture_postfix_update(
  %x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>, %expr: i32,
  %xf: !llvm.ptr<f32>, %vf: !llvm.ptr<f32>, %exprf: f32) -> () {
  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw sub i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.sub %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw and i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.and %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw or i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.or %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw xor i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.xor %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = mul i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.mul %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = sdiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.sdiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = udiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.udiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = shl i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.shl %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = lshr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.lshr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = ashr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.ashr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], i32* %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg i32* %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[xvalf:.*]] = bitcast i32 %[[xval]] to float
  // CHECK: %[[newval:.*]] = fadd float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], float* %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK: %[[xf_bitcast:.*]] = bitcast float* %[[xf]] to i32*
  // CHECK: %{{.*}} = cmpxchg i32* %[[xf_bitcast]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[xvalf]], float* %[[vf]]
  omp.atomic.capture {
    omp.atomic.read %vf = %xf : !llvm.ptr<f32>
    omp.atomic.update %xf : !llvm.ptr<f32> {
    ^bb0(%xval: f32):
      %newval = llvm.fadd %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[xvalf:.*]] = bitcast i32 %[[xval]] to float
  // CHECK: %[[newval:.*]] = fsub float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], float* %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK: %[[xf_bitcast:.*]] = bitcast float* %[[xf]] to i32*
  // CHECK: %{{.*}} = cmpxchg i32* %[[xf_bitcast]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[xvalf]], float* %[[vf]]
  omp.atomic.capture {
    omp.atomic.read %vf = %xf : !llvm.ptr<f32>
    omp.atomic.update %xf : !llvm.ptr<f32> {
    ^bb0(%xval: f32):
      %newval = llvm.fsub %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
  }

  llvm.return
}

// -----
// CHECK-LABEL: @omp_atomic_capture_misc
// CHECK-SAME: (i32* %[[x:.*]], i32* %[[v:.*]], i32 %[[expr:.*]], float* %[[xf:.*]], float* %[[vf:.*]], float %[[exprf:.*]])
llvm.func @omp_atomic_capture_misc(
  %x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>, %expr: i32,
  %xf: !llvm.ptr<f32>, %vf: !llvm.ptr<f32>, %exprf: f32) -> () {
  // CHECK: %[[xval:.*]] = atomicrmw xchg i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[xval]], i32* %[[v]]
  omp.atomic.capture{
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.write %x = %expr : !llvm.ptr<i32>, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[xvalf:.*]] = bitcast i32 %[[xval]] to float
  // CHECK: store float %[[exprf]], float* %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, i32* %{{.*}}
  // CHECK: %[[xf_bitcast:.*]] = bitcast float* %[[xf]] to i32*
  // CHECK: %{{.*}} = cmpxchg i32* %[[xf_bitcast]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[xvalf]], float* %[[vf]]
  omp.atomic.capture{
    omp.atomic.read %vf = %xf : !llvm.ptr<f32>
    omp.atomic.write %xf = %exprf : !llvm.ptr<f32>, f32
  }

  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] seq_cst
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] acquire
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture memory_order(acquire) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] release
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture memory_order(release) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add i32* %[[x]], i32 %[[expr]] acq_rel
  // CHECK: store i32 %[[res]], i32* %[[v]]
  omp.atomic.capture memory_order(acq_rel) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  llvm.return
}

// -----

// CHECK-LABEL: @omp_sections_empty
llvm.func @omp_sections_empty() -> () {
  omp.sections {
    omp.terminator
  }
  // CHECK-NEXT: ret void
  llvm.return
}

// -----

// Check IR generation for simple empty sections. This only checks the overall
// shape of the IR, detailed checking is done by the OpenMPIRBuilder.

// CHECK-LABEL: @omp_sections_trivial
llvm.func @omp_sections_trivial() -> () {
  // CHECK:   br label %[[ENTRY:[a-zA-Z_.]+]]

  // CHECK: [[ENTRY]]:
  // CHECK:   br label %[[PREHEADER:.*]]

  // CHECK: [[PREHEADER]]:
  // CHECK:   %{{.*}} = call i32 @__kmpc_global_thread_num({{.*}})
  // CHECK:   call void @__kmpc_for_static_init_4u({{.*}})
  // CHECK:   br label %[[HEADER:.*]]

  // CHECK: [[HEADER]]:
  // CHECK:   br label %[[COND:.*]]

  // CHECK: [[COND]]:
  // CHECK:   br i1 %{{.*}}, label %[[BODY:.*]], label %[[EXIT:.*]]
  // CHECK: [[BODY]]:
  // CHECK:   switch i32 %{{.*}}, label %[[INC:.*]] [
  // CHECK-NEXT:     i32 0, label %[[SECTION1:.*]]
  // CHECK-NEXT:     i32 1, label %[[SECTION2:.*]]
  // CHECK-NEXT: ]

  omp.sections {
    omp.section {
      // CHECK: [[SECTION1]]:
      // CHECK-NEXT: br label %[[SECTION1_REGION1:[^ ,]*]]
      // CHECK-EMPTY:
      // CHECK-NEXT: [[SECTION1_REGION1]]:
      // CHECK-NEXT: br label %[[SECTION1_REGION2:[^ ,]*]]
      // CHECK-EMPTY:
      // CHECK-NEXT: [[SECTION1_REGION2]]:
      // CHECK-NEXT: br label %[[INC]]
      omp.terminator
    }
    omp.section {
      // CHECK: [[SECTION2]]:
      // CHECK: br label %[[INC]]
      omp.terminator
    }
    omp.terminator
  }

  // CHECK: [[INC]]:
  // CHECK:   %{{.*}} = add {{.*}}, 1
  // CHECK:   br label %[[HEADER]]

  // CHECK: [[EXIT]]:
  // CHECK:   call void @__kmpc_for_static_fini({{.*}})
  // CHECK:   call void @__kmpc_barrier({{.*}})
  // CHECK:   br label %[[AFTER:.*]]

  // CHECK: [[AFTER]]:
  // CHECK:   ret void
  llvm.return
}

// -----

// CHECK: declare void @foo()
llvm.func @foo()

// CHECK: declare void @bar(i32)
llvm.func @bar(%arg0 : i32)

// CHECK-LABEL: @omp_sections
llvm.func @omp_sections(%arg0 : i32, %arg1 : i32, %arg2 : !llvm.ptr<i32>) -> () {

  // CHECK: switch i32 %{{.*}}, label %{{.*}} [
  // CHECK-NEXT:   i32 0, label %[[SECTION1:.*]]
  // CHECK-NEXT:   i32 1, label %[[SECTION2:.*]]
  // CHECK-NEXT:   i32 2, label %[[SECTION3:.*]]
  // CHECK-NEXT: ]
  omp.sections {
    omp.section {
      // CHECK: [[SECTION1]]:
      // CHECK:   br label %[[REGION1:[^ ,]*]]
      // CHECK: [[REGION1]]:
      // CHECK:   call void @foo()
      // CHECK:   br label %{{.*}}
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    omp.section {
      // CHECK: [[SECTION2]]:
      // CHECK:   br label %[[REGION2:[^ ,]*]]
      // CHECK: [[REGION2]]:
      // CHECK:   call void @bar(i32 %{{.*}})
      // CHECK:   br label %{{.*}}
      llvm.call @bar(%arg0) : (i32) -> ()
      omp.terminator
    }
    omp.section {
      // CHECK: [[SECTION3]]:
      // CHECK:   br label %[[REGION3:[^ ,]*]]
      // CHECK: [[REGION3]]:
      // CHECK:   %11 = add i32 %{{.*}}, %{{.*}}
      %add = llvm.add %arg0, %arg1 : i32
      // CHECK:   store i32 %{{.*}}, i32* %{{.*}}, align 4
      // CHECK:   br label %{{.*}}
      llvm.store %add, %arg2 : !llvm.ptr<i32>
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @foo()

// CHECK-LABEL: @omp_sections_with_clauses
llvm.func @omp_sections_with_clauses() -> () {
  // CHECK-NOT: call void @__kmpc_barrier
  omp.sections nowait {
    omp.section {
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    omp.section {
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Check that translation doesn't crash in presence of repeated successor
// blocks with different arguments within OpenMP operations: LLVM cannot
// represent this and a dummy block will be introduced for forwarding. The
// introduction mechanism itself is tested elsewhere.
// CHECK-LABEL: @repeated_successor
llvm.func @repeated_successor(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i1) {
  omp.wsloop for (%arg4) : i64 = (%arg0) to (%arg1) step (%arg2)  {
    llvm.cond_br %arg3, ^bb1(%arg0 : i64), ^bb1(%arg1 : i64)
  ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb0
    omp.yield
  }
  llvm.return
}

// -----

// CHECK-LABEL: @single
// CHECK-SAME: (i32 %[[x:.*]], i32 %[[y:.*]], i32* %[[zaddr:.*]])
llvm.func @single(%x: i32, %y: i32, %zaddr: !llvm.ptr<i32>) {
  // CHECK: %[[a:.*]] = sub i32 %[[x]], %[[y]]
  %a = llvm.sub %x, %y : i32
  // CHECK: store i32 %[[a]], i32* %[[zaddr]]
  llvm.store %a, %zaddr : !llvm.ptr<i32>
  // CHECK: call i32 @__kmpc_single
  omp.single {
    // CHECK: %[[z:.*]] = add i32 %[[x]], %[[y]]
    %z = llvm.add %x, %y : i32
    // CHECK: store i32 %[[z]], i32* %[[zaddr]]
    llvm.store %z, %zaddr : !llvm.ptr<i32>
    // CHECK: call void @__kmpc_end_single
    // CHECK: call void @__kmpc_barrier
    omp.terminator
  }
  // CHECK: %[[b:.*]] = mul i32 %[[x]], %[[y]]
  %b = llvm.mul %x, %y : i32
  // CHECK: store i32 %[[b]], i32* %[[zaddr]]
  llvm.store %b, %zaddr : !llvm.ptr<i32>
  // CHECK: ret void
  llvm.return
}

// -----

// CHECK-LABEL: @single_nowait
// CHECK-SAME: (i32 %[[x:.*]], i32 %[[y:.*]], i32* %[[zaddr:.*]])
llvm.func @single_nowait(%x: i32, %y: i32, %zaddr: !llvm.ptr<i32>) {
  // CHECK: %[[a:.*]] = sub i32 %[[x]], %[[y]]
  %a = llvm.sub %x, %y : i32
  // CHECK: store i32 %[[a]], i32* %[[zaddr]]
  llvm.store %a, %zaddr : !llvm.ptr<i32>
  // CHECK: call i32 @__kmpc_single
  omp.single nowait {
    // CHECK: %[[z:.*]] = add i32 %[[x]], %[[y]]
    %z = llvm.add %x, %y : i32
    // CHECK: store i32 %[[z]], i32* %[[zaddr]]
    llvm.store %z, %zaddr : !llvm.ptr<i32>
    // CHECK: call void @__kmpc_end_single
    // CHECK-NOT: call void @__kmpc_barrier
    omp.terminator
  }
  // CHECK: %[[t:.*]] = mul i32 %[[x]], %[[y]]
  %t = llvm.mul %x, %y : i32
  // CHECK: store i32 %[[t]], i32* %[[zaddr]]
  llvm.store %t, %zaddr : !llvm.ptr<i32>
  // CHECK: ret void
  llvm.return
}

// -----

// CHECK: @_QFsubEx = internal global i32 undef
// CHECK: @_QFsubEx.cache = common global i8** null

// CHECK-LABEL: @omp_threadprivate
llvm.func @omp_threadprivate() {
// CHECK:  [[THREAD:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB:[0-9]+]])
// CHECK:  [[TMP1:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB]], i32 [[THREAD]], i8* bitcast (i32* @_QFsubEx to i8*), i64 4, i8*** @_QFsubEx.cache)
// CHECK:  [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK:  store i32 1, i32* [[TMP2]], align 4
// CHECK:  store i32 3, i32* [[TMP2]], align 4

// CHECK-LABEL: omp.par.region{{.*}}
// CHECK:  [[THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB2:[0-9]+]])
// CHECK:  [[TMP3:%.*]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* @[[GLOB2]], i32 [[THREAD2]], i8* bitcast (i32* @_QFsubEx to i8*), i64 4, i8*** @_QFsubEx.cache)
// CHECK:  [[TMP4:%.*]] = bitcast i8* [[TMP3]] to i32*
// CHECK:  store i32 2, i32* [[TMP4]], align 4

  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.mlir.constant(3 : i32) : i32

  %3 = llvm.mlir.addressof @_QFsubEx : !llvm.ptr<i32>
  %4 = omp.threadprivate %3 : !llvm.ptr<i32> -> !llvm.ptr<i32>

  llvm.store %0, %4 : !llvm.ptr<i32>

  omp.parallel  {
    %5 = omp.threadprivate %3 : !llvm.ptr<i32> -> !llvm.ptr<i32>
    llvm.store %1, %5 : !llvm.ptr<i32>
    omp.terminator
  }

  llvm.store %2, %4 : !llvm.ptr<i32>
  llvm.return
}

llvm.mlir.global internal @_QFsubEx() : i32
