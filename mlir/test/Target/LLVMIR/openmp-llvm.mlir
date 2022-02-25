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
 omp.wsloop (%iv) : i32 = (%lb) to (%ub) step (%step) schedule(static) {
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
 omp.wsloop (%iv) : i32 = (%lb) to (%ub) step (%step) schedule(static = %static_chunk_size : i32) {
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
 omp.wsloop (%iv) : i32 = (%lb) to (%ub) step (%step) schedule(static = %static_chunk_size : i32) {
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
 omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic) {
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
 omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic = %chunk_size_const : i16) {
  // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 35, i64 {{.*}}, i64 %{{.*}}, i64 {{.*}}, i64 2)
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
 omp.wsloop (%iv) : i32 = (%lb) to (%ub) step (%step) schedule(dynamic = %chunk_size_var : i16) {
  // CHECK: %[[CHUNK_SIZE:.*]] = sext i16 %{{.*}} to i32
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 35, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %[[CHUNK_SIZE]])
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
 omp.wsloop (%iv) : i32 = (%lb) to (%ub) step (%step) schedule(dynamic = %chunk_size_var : i64) {
  // CHECK: %[[CHUNK_SIZE:.*]] = trunc i64 %{{.*}} to i32
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 35, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %[[CHUNK_SIZE]])
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
 omp.wsloop (%iv) : i32 = (%lb) to (%ub) step (%step) schedule(dynamic = %chunk_size : i32) {
  // CHECK: call void @__kmpc_dispatch_init_4u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 35, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %{{.*}})
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
 omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(auto) {
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
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(runtime) {
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
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(guided) {
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
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, nonmonotonic) {
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
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, monotonic) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 536870947
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

llvm.func @test_omp_wsloop_runtime_simd(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(runtime, simd) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 47
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

llvm.func @test_omp_wsloop_guided_simd(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(guided, simd) {
    // CHECK: call void @__kmpc_dispatch_init_8u(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 46
    // CHECK: %[[continue:.*]] = call i32 @__kmpc_dispatch_next_8u
    // CHECK: %[[cond:.*]] = icmp ne i32 %[[continue]], 0
    // CHECK  br i1 %[[cond]], label %omp_loop.header{{.*}}, label %omp_loop.exit{{.*}}
    llvm.call @body(%iv) : (i64) -> ()
    omp.yield
  }
  llvm.return
}

// -----

omp.critical.declare @mutex hint(contended)

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

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex.var{{.*}}, i32 2)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex) {
  // CHECK: store
    llvm.store %xval, %x : !llvm.ptr<i32>
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex.var{{.*}})
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
    omp.wsloop (%arg0, %arg1, %arg2) : i32 = (%0, %1, %2) to (%3, %4, %5) step (%6, %7, %8) collapse(3) {
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

  omp.wsloop (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) ordered(0) {
    // CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @[[GLOB1:[0-9]+]])
    // CHECK-NEXT:  call void @__kmpc_ordered(%struct.ident_t* @[[GLOB1]], i32 [[OMP_THREAD]])
    omp.ordered_region  {
      omp.terminator
    // CHECK: call void @__kmpc_end_ordered(%struct.ident_t* @[[GLOB1]], i32 [[OMP_THREAD]])
    }
    omp.yield
  }

  omp.wsloop (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) ordered(1) {
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

  omp.wsloop (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) ordered(2) {
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

  // CHECK: [[INC]]:
  // CHECK:   %{{.*}} = add {{.*}}, 1
  // CHECK:   br label %[[HEADER]]

  // CHECK: [[EXIT]]:
  // CHECK:   call void @__kmpc_for_static_fini({{.*}})
  // CHECK:   call void @__kmpc_barrier({{.*}})
  // CHECK:   br label %[[AFTER:.*]]

  // CHECK: [[AFTER]]:
  // CHECK:   br label %[[END:.*]]

  // CHECK: [[END]]:
  // CHECK:   ret void
  omp.sections {
    omp.section {
      // CHECK: [[SECTION1]]:
      // CHECK-NEXT: br label %[[REGION1:[^ ,]*]]
      // CHECK: [[REGION1]]:
      // CHECK-NEXT: br label %[[EXIT]]
      omp.terminator
    }
    omp.section {
      // CHECK: [[SECTION2]]:
      // CHECK-NEXT: br label %[[REGION2:[^ ,]*]]
      // CHECK: [[REGION2]]:
      // CHECK-NEXT: br label %[[EXIT]]
      omp.terminator
    }
    omp.terminator
  }
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
  omp.wsloop (%arg4) : i64 = (%arg0) to (%arg1) step (%arg2)  {
    llvm.cond_br %arg3, ^bb1(%arg0 : i64), ^bb1(%arg1 : i64)
  ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb0
    omp.yield
  }
  llvm.return
}
