// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_stand_alone_directives()
llvm.func @test_stand_alone_directives() {
  // CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
  // CHECK-NEXT:  call void @__kmpc_barrier(ptr @{{[0-9]+}}, i32 [[OMP_THREAD]])
  omp.barrier

  // CHECK: [[OMP_THREAD1:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
  // CHECK-NEXT:  [[RET_VAL:%.*]] = call i32 @__kmpc_omp_taskwait(ptr @{{[0-9]+}}, i32 [[OMP_THREAD1]])
  omp.taskwait

  // CHECK: [[OMP_THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
  // CHECK-NEXT:  [[RET_VAL:%.*]] = call i32 @__kmpc_omp_taskyield(ptr @{{[0-9]+}}, i32 [[OMP_THREAD2]], i32 0)
  omp.taskyield

  // CHECK-NEXT:    ret void
  llvm.return
}

// CHECK-LABEL: define void @test_flush_construct(ptr %{{[0-9]+}})
llvm.func @test_flush_construct(%arg0: !llvm.ptr) {
  // CHECK: call void @__kmpc_flush(ptr @{{[0-9]+}}
  omp.flush

  // CHECK: call void @__kmpc_flush(ptr @{{[0-9]+}}
  omp.flush (%arg0 : !llvm.ptr)

  // CHECK: call void @__kmpc_flush(ptr @{{[0-9]+}}
  omp.flush (%arg0, %arg0 : !llvm.ptr, !llvm.ptr)

  %0 = llvm.mlir.constant(1 : i64) : i64
  //  CHECK: alloca {{.*}} align 4
  %1 = llvm.alloca %0 x i32 {in_type = i32, name = "a"} : (i64) -> !llvm.ptr
  // CHECK: call void @__kmpc_flush(ptr @{{[0-9]+}}
  omp.flush
  //  CHECK: load i32, ptr
  %2 = llvm.load %1 : !llvm.ptr -> i32

  // CHECK-NEXT:    ret void
  llvm.return
}

// CHECK-LABEL: define void @test_omp_parallel_1()
llvm.func @test_omp_parallel_1() -> () {
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_1:.*]])
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
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_2:.*]])
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
  // CHECK: %[[GTN_NUM_THREADS_VAR_1:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GTN_SI_VAR_1:.*]])
  // CHECK: call void @__kmpc_push_num_threads(ptr @[[GTN_SI_VAR_1]], i32 %[[GTN_NUM_THREADS_VAR_1]], i32 %[[NUM_THREADS_VAR_1]])
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_1:.*]])
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
  // CHECK: %[[GTN_NUM_THREADS_VAR_2:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GTN_SI_VAR_2:.*]])
  // CHECK: call void @__kmpc_push_num_threads(ptr @[[GTN_SI_VAR_2]], i32 %[[GTN_NUM_THREADS_VAR_2]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_2:.*]])
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
  // CHECK: %[[GTN_NUM_THREADS_VAR_3_1:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GTN_SI_VAR_3_1:.*]])
  // CHECK: call void @__kmpc_push_num_threads(ptr @[[GTN_SI_VAR_3_1]], i32 %[[GTN_NUM_THREADS_VAR_3_1]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_3_1:.*]])
  omp.parallel num_threads(%0: i32) {
    omp.barrier
    omp.terminator
  }
  %1 = llvm.mlir.constant(8 : index) : i32
  // CHECK: %[[GTN_NUM_THREADS_VAR_3_2:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GTN_SI_VAR_3_2:.*]])
  // CHECK: call void @__kmpc_push_num_threads(ptr @[[GTN_SI_VAR_3_2]], i32 %[[GTN_NUM_THREADS_VAR_3_2]], i32 8)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_NUM_THREADS_3_2:.*]])
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

  %0 = llvm.mlir.constant(0 : index) : i32
  %1 = llvm.icmp "slt" %arg0, %0 : i32
// CHECK: %[[IF_COND_VAR_1:.*]] = icmp slt i32 %[[IF_VAR_1]], 0


// CHECK: %[[GTN_IF_1:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[SI_VAR_IF_1:.*]])
// CHECK: br label %[[OUTLINED_CALL_IF_BLOCK_1:.*]]
// CHECK: [[OUTLINED_CALL_IF_BLOCK_1]]:
// CHECK: %[[I32_IF_COND_VAR_1:.*]] = sext i1 %[[IF_COND_VAR_1]] to i32
// CHECK: call void @__kmpc_fork_call_if(ptr @[[SI_VAR_IF_1]], i32 0, ptr @[[OMP_OUTLINED_FN_IF_1:.*]], i32 %[[I32_IF_COND_VAR_1]], ptr null)
// CHECK: br label %[[OUTLINED_EXIT_IF_1:.*]]
// CHECK: [[OUTLINED_EXIT_IF_1]]:
// CHECK: br label %[[RETURN_BLOCK_IF_1:.*]]
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

// CHECK-LABEL: define void @test_omp_parallel_attrs()
llvm.func @test_omp_parallel_attrs() -> () attributes {
  target_cpu = "x86-64",
  target_features = #llvm.target_features<["+mmx", "+sse"]>
} {
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN:.*]])
  omp.parallel {
    omp.barrier
    omp.terminator
  }

  llvm.return
}

// CHECK: define {{.*}} @[[OMP_OUTLINED_FN]]{{.*}} #[[ATTRS:[0-9]+]]
// CHECK: attributes #[[ATTRS]] = {
// CHECK-SAME: "target-cpu"="x86-64"
// CHECK-SAME: "target-features"="+mmx,+sse"

// -----

// CHECK-LABEL: define void @test_omp_parallel_3()
llvm.func @test_omp_parallel_3() -> () {
  // CHECK: [[OMP_THREAD_3_1:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
  // CHECK: call void @__kmpc_push_proc_bind(ptr @{{[0-9]+}}, i32 [[OMP_THREAD_3_1]], i32 2)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_3_1:.*]])
  omp.parallel proc_bind(master) {
    omp.barrier
    omp.terminator
  }
  // CHECK: [[OMP_THREAD_3_2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
  // CHECK: call void @__kmpc_push_proc_bind(ptr @{{[0-9]+}}, i32 [[OMP_THREAD_3_2]], i32 3)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_3_2:.*]])
  omp.parallel proc_bind(close) {
    omp.barrier
    omp.terminator
  }
  // CHECK: [[OMP_THREAD_3_3:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
  // CHECK: call void @__kmpc_push_proc_bind(ptr @{{[0-9]+}}, i32 [[OMP_THREAD_3_3]], i32 4)
  // CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_3_3:.*]])
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
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_4_1:.*]])
// CHECK: define internal void @[[OMP_OUTLINED_FN_4_1]]
// CHECK: call void @__kmpc_barrier
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_4_1_1:.*]])
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
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_5_1:.*]])
// CHECK: define internal void @[[OMP_OUTLINED_FN_5_1]]
// CHECK: call void @__kmpc_barrier
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_5_1_1:.*]])
// CHECK: call void @__kmpc_barrier
  omp.parallel {
    omp.barrier

// CHECK: define internal void @[[OMP_OUTLINED_FN_5_1_1]]
    omp.parallel {
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @[[OMP_OUTLINED_FN_5_1_1_1:.*]])
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
// CHECK: call void {{.*}}@__kmpc_fork_call{{.*}} @{{.*}})
// CHECK: omp.par.region1:
  omp.parallel {
    omp.master {
// CHECK: [[OMP_THREAD_3_4:%.*]] = call i32 @__kmpc_global_thread_num(ptr @{{[0-9]+}})
// CHECK: {{[0-9]+}} = call i32 @__kmpc_master(ptr @{{[0-9]+}}, i32 [[OMP_THREAD_3_4]])
// CHECK: omp.master.region
// CHECK: call void @__kmpc_end_master(ptr @{{[0-9]+}}, i32 [[OMP_THREAD_3_4]])
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
// CHECK: @[[$loc:.*]] = private unnamed_addr constant {{.*}} c";unknown;unknown;{{[0-9]+}};{{[0-9]+}};;\00"
// CHECK: @[[$loc_struct:.*]] = private unnamed_addr constant %struct.ident_t {{.*}} @[[$loc]] {{.*}}

// CHECK-LABEL: @wsloop_simple
llvm.func @wsloop_simple(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  omp.parallel {
    "omp.wsloop"(%1, %0, %2) ({
    ^bb0(%arg1: i64):
      // The form of the emitted IR is controlled by OpenMPIRBuilder and
      // tested there. Just check that the right functions are called.
      // CHECK: call i32 @__kmpc_global_thread_num
      // CHECK: call void @__kmpc_for_static_init_{{.*}}(ptr @[[$loc_struct]],
      %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
      %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %3, %4 : f32, !llvm.ptr
      omp.yield
      // CHECK: call void @__kmpc_for_static_fini(ptr @[[$loc_struct]],
    }) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (i64, i64, i64) -> ()
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL: @wsloop_inclusive_1
llvm.func @wsloop_inclusive_1(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  // CHECK: store i64 31, ptr %{{.*}}upperbound
  "omp.wsloop"(%1, %0, %2) ({
  ^bb0(%arg1: i64):
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %4 : f32, !llvm.ptr
    omp.yield
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (i64, i64, i64) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @wsloop_inclusive_2
llvm.func @wsloop_inclusive_2(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  // CHECK: store i64 32, ptr %{{.*}}upperbound
  "omp.wsloop"(%1, %0, %2) ({
  ^bb0(%arg1: i64):
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %4 : f32, !llvm.ptr
    omp.yield
  }) {inclusive, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (i64, i64, i64) -> ()
  llvm.return
}

// -----

llvm.func @body(i32)

// CHECK-LABEL: @test_omp_wsloop_static_defchunk
llvm.func @test_omp_wsloop_static_defchunk(%lb : i32, %ub : i32, %step : i32) -> () {
 omp.wsloop schedule(static)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
   // CHECK: call void @__kmpc_for_static_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 34, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 1, i32 0)
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
   // CHECK: call void @__kmpc_for_static_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 33, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 1, i32 1)
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
   // CHECK: call void @__kmpc_for_static_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 33, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 1, i32 2)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741859, i64 {{.*}}, i64 %{{.*}}, i64 {{.*}}, i64 2)
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
 %chunk_size_alloca = llvm.alloca %1 x i16 {bindc_name = "chunk_size", in_type = i16, uniq_name = "_QFsub1Echunk_size"} : (i64) -> !llvm.ptr
 %chunk_size_var = llvm.load %chunk_size_alloca : !llvm.ptr -> i16
 omp.wsloop schedule(dynamic = %chunk_size_var : i16)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
  // CHECK: %[[CHUNK_SIZE:.*]] = sext i16 %{{.*}} to i32
  // CHECK: call void @__kmpc_dispatch_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741859, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %[[CHUNK_SIZE]])
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
 %chunk_size_alloca = llvm.alloca %1 x i64 {bindc_name = "chunk_size", in_type = i64, uniq_name = "_QFsub1Echunk_size"} : (i64) -> !llvm.ptr
 %chunk_size_var = llvm.load %chunk_size_alloca : !llvm.ptr -> i64
 omp.wsloop schedule(dynamic = %chunk_size_var : i64)
 for (%iv) : i32 = (%lb) to (%ub) step (%step) {
  // CHECK: %[[CHUNK_SIZE:.*]] = trunc i64 %{{.*}} to i32
  // CHECK: call void @__kmpc_dispatch_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741859, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %[[CHUNK_SIZE]])
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
  // CHECK: call void @__kmpc_dispatch_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741859, i32 {{.*}}, i32 %{{.*}}, i32 {{.*}}, i32 %{{.*}})
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
    // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741859
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
    // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 536870947
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
    // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741871
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
    // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741870
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
llvm.func @simdloop_simple(%lb : i64, %ub : i64, %step : i64, %arg0: !llvm.ptr) {
  "omp.simdloop" (%lb, %ub, %step) ({
    ^bb0(%iv: i64):
      %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
      // The form of the emitted IR is controlled by OpenMPIRBuilder and
      // tested there. Just check that the right metadata is added.
      // CHECK: llvm.access.group
      %4 = llvm.getelementptr %arg0[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %3, %4 : f32, !llvm.ptr
      omp.yield
  }) {operandSegmentSizes = array<i32: 1,1,1,0,0,0>} :
    (i64, i64, i64) -> ()

  llvm.return
}
// CHECK: llvm.loop.parallel_accesses
// CHECK-NEXT: llvm.loop.vectorize.enable

// -----

// CHECK-LABEL: @simdloop_simple_multiple
llvm.func @simdloop_simple_multiple(%lb1 : i64, %ub1 : i64, %step1 : i64, %lb2 : i64, %ub2 : i64, %step2 : i64, %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  omp.simdloop for (%iv1, %iv2) : i64 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    // The form of the emitted IR is controlled by OpenMPIRBuilder and
    // tested there. Just check that the right metadata is added and collapsed
    // loop bound is generated (Collapse clause is represented as a loop with
    // list of indices, bounds and steps where the size of the list is equal
    // to the collapse value.)
    // CHECK: icmp slt i64
    // CHECK-COUNT-3: select
    // CHECK: %[[TRIPCOUNT0:.*]] = select
    // CHECK: br label %[[PREHEADER:.*]]
    // CHECK: [[PREHEADER]]:
    // CHECK: icmp slt i64
    // CHECK-COUNT-3: select
    // CHECK: %[[TRIPCOUNT1:.*]] = select
    // CHECK: mul nuw i64 %[[TRIPCOUNT0]], %[[TRIPCOUNT1]]
    // CHECK: br label %[[COLLAPSED_PREHEADER:.*]]
    // CHECK: [[COLLAPSED_PREHEADER]]:
    // CHECK: br label %[[COLLAPSED_HEADER:.*]]
    // CHECK: llvm.access.group
    // CHECK-NEXT: llvm.access.group
    %4 = llvm.getelementptr %arg0[%iv1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %5 = llvm.getelementptr %arg1[%iv2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %4 : f32, !llvm.ptr
    llvm.store %3, %5 : f32, !llvm.ptr
    omp.yield
  }
  llvm.return
}
// CHECK: llvm.loop.parallel_accesses
// CHECK-NEXT: llvm.loop.vectorize.enable

// -----

// CHECK-LABEL: @simdloop_simple_multiple_simdlen
llvm.func @simdloop_simple_multiple_simdlen(%lb1 : i64, %ub1 : i64, %step1 : i64, %lb2 : i64, %ub2 : i64, %step2 : i64, %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  omp.simdloop simdlen(2) for (%iv1, %iv2) : i64 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    // The form of the emitted IR is controlled by OpenMPIRBuilder and
    // tested there. Just check that the right metadata is added.
    // CHECK: llvm.access.group
    // CHECK-NEXT: llvm.access.group
    %4 = llvm.getelementptr %arg0[%iv1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %5 = llvm.getelementptr %arg1[%iv2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %4 : f32, !llvm.ptr
    llvm.store %3, %5 : f32, !llvm.ptr
    omp.yield
  }
  llvm.return
}
// CHECK: llvm.loop.parallel_accesses
// CHECK-NEXT: llvm.loop.vectorize.enable
// CHECK-NEXT: llvm.loop.vectorize.width{{.*}}i64 2

// -----

// CHECK-LABEL: @simdloop_simple_multiple_safelen
llvm.func @simdloop_simple_multiple_safelen(%lb1 : i64, %ub1 : i64, %step1 : i64, %lb2 : i64, %ub2 : i64, %step2 : i64, %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  omp.simdloop safelen(2) for (%iv1, %iv2) : i64 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%iv1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %5 = llvm.getelementptr %arg1[%iv2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %4 : f32, !llvm.ptr
    llvm.store %3, %5 : f32, !llvm.ptr
    omp.yield
  }
  llvm.return
}
// CHECK: llvm.loop.vectorize.enable
// CHECK-NEXT: llvm.loop.vectorize.width{{.*}}i64 2

// -----

// CHECK-LABEL: @simdloop_simple_multiple_simdlen_safelen
llvm.func @simdloop_simple_multiple_simdlen_safelen(%lb1 : i64, %ub1 : i64, %step1 : i64, %lb2 : i64, %ub2 : i64, %step2 : i64, %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  omp.simdloop simdlen(1) safelen(2) for (%iv1, %iv2) : i64 = (%lb1, %lb2) to (%ub1, %ub2) step (%step1, %step2) {
    %3 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %4 = llvm.getelementptr %arg0[%iv1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %5 = llvm.getelementptr %arg1[%iv2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %4 : f32, !llvm.ptr
    llvm.store %3, %5 : f32, !llvm.ptr
    omp.yield
  }
  llvm.return
}
// CHECK: llvm.loop.vectorize.enable
// CHECK-NEXT: llvm.loop.vectorize.width{{.*}}i64 1

// -----

// CHECK-LABEL: @simdloop_if
llvm.func @simdloop_if(%arg0: !llvm.ptr {fir.bindc_name = "n"}, %arg1: !llvm.ptr {fir.bindc_name = "threshold"}) {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {adapt.valuebyref, in_type = i32, operandSegmentSizes = array<i32: 0, 0>} : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "i", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFtest_simdEi"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(0 : i32) : i32
  %5 = llvm.load %arg0 : !llvm.ptr -> i32
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.load %arg0 : !llvm.ptr -> i32
  %8 = llvm.load %arg1 : !llvm.ptr -> i32
  %9 = llvm.icmp "sge" %7, %8 : i32
  omp.simdloop   if(%9) for  (%arg2) : i32 = (%4) to (%5) inclusive step (%6) {
    // The form of the emitted IR is controlled by OpenMPIRBuilder and
    // tested there. Just check that the right metadata is added.
    // CHECK: llvm.access.group
    llvm.store %arg2, %1 : i32, !llvm.ptr
    omp.yield
  }
  llvm.return
}
// Be sure that llvm.loop.vectorize.enable metadata appears twice
// CHECK: llvm.loop.parallel_accesses
// CHECK-NEXT: llvm.loop.vectorize.enable
// CHECK: llvm.loop.vectorize.enable

// -----

llvm.func @body(i64)

llvm.func @test_omp_wsloop_ordered(%lb : i64, %ub : i64, %step : i64) -> () {
 omp.wsloop ordered(0)
 for (%iv) : i64 = (%lb) to (%ub) step (%step) {
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 66, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 66, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_4u(ptr @{{.*}}, i32 %{{.*}}, i32 65, i32 1, i32 %{{.*}}, i32 1, i32 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 67, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 70, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 69, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 68, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 1073741891, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
  // CHECK: call void @__kmpc_dispatch_init_8u(ptr @{{.*}}, i32 %{{.*}}, i32 536870979, i64 1, i64 %{{.*}}, i64 1, i64 1)
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
llvm.func @omp_critical(%x : !llvm.ptr, %xval : i32) -> () {
  // CHECK: call void @__kmpc_critical({{.*}}critical_user_.var{{.*}})
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_none.var{{.*}}, i32 0)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_none) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_none.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_uncontended.var{{.*}}, i32 1)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_uncontended) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_uncontended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_contended.var{{.*}}, i32 2)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_contended) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_contended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_nonspeculative.var{{.*}}, i32 4)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_nonspeculative) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_nonspeculative.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_nonspeculative_uncontended.var{{.*}}, i32 5)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_nonspeculative_uncontended) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_nonspeculative_uncontended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_nonspeculative_contended.var{{.*}}, i32 6)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_nonspeculative_contended) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_nonspeculative_contended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_speculative.var{{.*}}, i32 8)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_speculative) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_speculative.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_speculative_uncontended.var{{.*}}, i32 9)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_speculative_uncontended) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
    omp.terminator
  }
  // CHECK: call void @__kmpc_end_critical({{.*}}critical_user_mutex_speculative_uncontended.var{{.*}})

  // CHECK: call void @__kmpc_critical_with_hint({{.*}}critical_user_mutex_speculative_contended.var{{.*}}, i32 10)
  // CHECK: br label %omp.critical.region
  // CHECK: omp.critical.region
  omp.critical(@mutex_speculative_contended) {
  // CHECK: store
    llvm.store %xval, %x : i32, !llvm.ptr
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
// CHECK: ptr noalias %[[TIDADDR:[0-9A-Za-z.]*]]
// CHECK: load i32, ptr %[[TIDADDR]]
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
    %20: !llvm.ptr) {
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
    // CHECK: store i32 0, ptr
    // CHECK: %[[TOTAL_SUB_1:.*]] = sub i32 %[[TOTAL]], 1
    // CHECK: store i32 %[[TOTAL_SUB_1]], ptr
    // CHECK: call void @__kmpc_for_static_init_4u
    omp.wsloop
    for (%arg0, %arg1, %arg2) : i32 = (%0, %1, %2) to (%3, %4, %5) step (%6, %7, %8) {
      %31 = llvm.load %20 : !llvm.ptr -> i32
      %32 = llvm.add %31, %arg0 : i32
      %33 = llvm.add %32, %arg1 : i32
      %34 = llvm.add %33, %arg2 : i32
      llvm.store %34, %20 : i32, !llvm.ptr
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
// CHECK: ptr noalias %[[TIDADDR:[0-9A-Za-z.]*]]
// CHECK: load i32, ptr %[[TIDADDR]]
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
    %20: !llvm.ptr) {
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
    // CHECK: store i32 1, ptr
    // CHECK: store i32 %[[TOTAL]], ptr
    // CHECK: call void @__kmpc_dispatch_init_4u
    omp.wsloop schedule(dynamic)
    for (%arg0, %arg1, %arg2) : i32 = (%0, %1, %2) to (%3, %4, %5) step (%6, %7, %8) {
      %31 = llvm.load %20 : !llvm.ptr -> i32
      %32 = llvm.add %31, %arg0 : i32
      %33 = llvm.add %32, %arg1 : i32
      %34 = llvm.add %33, %arg2 : i32
      llvm.store %34, %20 : i32, !llvm.ptr
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

  // CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1:[0-9]+]])
  // CHECK-NEXT:  call void @__kmpc_ordered(ptr @[[GLOB1]], i32 [[OMP_THREAD]])
  omp.ordered_region {
    omp.terminator
  // CHECK: call void @__kmpc_end_ordered(ptr @[[GLOB1]], i32 [[OMP_THREAD]])
  }

  omp.wsloop ordered(0)
  for (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) {
    // CHECK:  call void @__kmpc_ordered(ptr @[[GLOB3:[0-9]+]], i32 [[OMP_THREAD2:%.*]])
    omp.ordered_region  {
      omp.terminator
    // CHECK: call void @__kmpc_end_ordered(ptr @[[GLOB3]], i32 [[OMP_THREAD2]])
    }
    omp.yield
  }

  omp.wsloop ordered(1)
  for (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) {
    // CHECK: [[TMP:%.*]] = getelementptr inbounds [1 x i64], ptr [[ADDR]], i64 0, i64 0
    // CHECK: store i64 [[ARG0:%.*]], ptr [[TMP]], align 8
    // CHECK: [[TMP2:%.*]] = getelementptr inbounds [1 x i64], ptr [[ADDR]], i64 0, i64 0
    // CHECK: [[OMP_THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB3:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_wait(ptr @[[GLOB3]], i32 [[OMP_THREAD2]], ptr [[TMP2]])
    omp.ordered depend_type(dependsink) depend_vec(%arg3 : i64) {num_loops_val = 1 : i64}

    // CHECK: [[TMP3:%.*]] = getelementptr inbounds [1 x i64], ptr [[ADDR3]], i64 0, i64 0
    // CHECK: store i64 [[ARG0]], ptr [[TMP3]], align 8
    // CHECK: [[TMP4:%.*]] = getelementptr inbounds [1 x i64], ptr [[ADDR3]], i64 0, i64 0
    // CHECK: [[OMP_THREAD4:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB5:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_post(ptr @[[GLOB5]], i32 [[OMP_THREAD4]], ptr [[TMP4]])
    omp.ordered depend_type(dependsource) depend_vec(%arg3 : i64) {num_loops_val = 1 : i64}

    omp.yield
  }

  omp.wsloop ordered(2)
  for (%arg7) : i32 = (%arg0) to (%arg1) step (%arg2) {
    // CHECK: [[TMP5:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR5]], i64 0, i64 0
    // CHECK: store i64 [[ARG0]], ptr [[TMP5]], align 8
    // CHECK: [[TMP6:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR5]], i64 0, i64 1
    // CHECK: store i64 [[ARG1:%.*]], ptr [[TMP6]], align 8
    // CHECK: [[TMP7:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR5]], i64 0, i64 0
    // CHECK: [[OMP_THREAD6:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB7:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_wait(ptr @[[GLOB7]], i32 [[OMP_THREAD6]], ptr [[TMP7]])
    // CHECK: [[TMP8:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR7]], i64 0, i64 0
    // CHECK: store i64 [[ARG2:%.*]], ptr [[TMP8]], align 8
    // CHECK: [[TMP9:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR7]], i64 0, i64 1
    // CHECK: store i64 [[ARG3:%.*]], ptr [[TMP9]], align 8
    // CHECK: [[TMP10:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR7]], i64 0, i64 0
    // CHECK: [[OMP_THREAD8:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB7]])
    // CHECK: call void @__kmpc_doacross_wait(ptr @[[GLOB7]], i32 [[OMP_THREAD8]], ptr [[TMP10]])
    omp.ordered depend_type(dependsink) depend_vec(%arg3, %arg4, %arg5, %arg6 : i64, i64, i64, i64) {num_loops_val = 2 : i64}

    // CHECK: [[TMP11:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR9]], i64 0, i64 0
    // CHECK: store i64 [[ARG0]], ptr [[TMP11]], align 8
    // CHECK: [[TMP12:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR9]], i64 0, i64 1
    // CHECK: store i64 [[ARG1]], ptr [[TMP12]], align 8
    // CHECK: [[TMP13:%.*]] = getelementptr inbounds [2 x i64], ptr [[ADDR9]], i64 0, i64 0
    // CHECK: [[OMP_THREAD10:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB9:[0-9]+]])
    // CHECK: call void @__kmpc_doacross_post(ptr @[[GLOB9]], i32 [[OMP_THREAD10]], ptr [[TMP13]])
    omp.ordered depend_type(dependsource) depend_vec(%arg3, %arg4 : i64, i64) {num_loops_val = 2 : i64}

    omp.yield
  }

  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_read
// CHECK-SAME: (ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
llvm.func @omp_atomic_read(%arg0 : !llvm.ptr, %arg1 : !llvm.ptr) -> () {

  // CHECK: %[[X1:.*]] = load atomic i32, ptr %[[ARG0]] monotonic, align 4
  // CHECK: store i32 %[[X1]], ptr %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 : !llvm.ptr, i32

  // CHECK: %[[X2:.*]] = load atomic i32, ptr %[[ARG0]] seq_cst, align 4
  // CHECK: call void @__kmpc_flush(ptr @{{.*}})
  // CHECK: store i32 %[[X2]], ptr %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 memory_order(seq_cst) : !llvm.ptr, i32

  // CHECK: %[[X3:.*]] = load atomic i32, ptr %[[ARG0]] acquire, align 4
  // CHECK: call void @__kmpc_flush(ptr @{{.*}})
  // CHECK: store i32 %[[X3]], ptr %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 memory_order(acquire) : !llvm.ptr, i32

  // CHECK: %[[X4:.*]] = load atomic i32, ptr %[[ARG0]] monotonic, align 4
  // CHECK: store i32 %[[X4]], ptr %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 memory_order(relaxed) : !llvm.ptr, i32
  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_write
// CHECK-SAME: (ptr %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_write(%x: !llvm.ptr, %expr: i32) -> () {
  // CHECK: store atomic i32 %[[expr]], ptr %[[x]] monotonic, align 4
  omp.atomic.write %x = %expr : !llvm.ptr, i32
  // CHECK: store atomic i32 %[[expr]], ptr %[[x]] seq_cst, align 4
  // CHECK: call void @__kmpc_flush(ptr @{{.*}})
  omp.atomic.write %x = %expr memory_order(seq_cst) : !llvm.ptr, i32
  // CHECK: store atomic i32 %[[expr]], ptr %[[x]] release, align 4
  // CHECK: call void @__kmpc_flush(ptr @{{.*}})
  omp.atomic.write %x = %expr memory_order(release) : !llvm.ptr, i32
  // CHECK: store atomic i32 %[[expr]], ptr %[[x]] monotonic, align 4
  omp.atomic.write %x = %expr memory_order(relaxed) : !llvm.ptr, i32
  llvm.return
}

// -----

// Checking simple atomicrmw and cmpxchg based translation. This also checks for
// ambigous alloca insert point by putting llvm.mul as the first update operation.
// CHECK-LABEL: @omp_atomic_update
// CHECK-SAME: (ptr %[[x:.*]], i32 %[[expr:.*]], ptr %[[xbool:.*]], i1 %[[exprbool:.*]])
llvm.func @omp_atomic_update(%x:!llvm.ptr, %expr: i32, %xbool: !llvm.ptr, %exprbool: i1) {
  // CHECK: %[[t1:.*]] = mul i32 %[[x_old:.*]], %[[expr]]
  // CHECK: store i32 %[[t1]], ptr %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, ptr %[[x_new]]
  // CHECK: cmpxchg ptr %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr {
  ^bb0(%xval: i32):
    %newval = llvm.mul %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  // CHECK: atomicrmw add ptr %[[x]], i32 %[[expr]] monotonic
  omp.atomic.update %x : !llvm.ptr {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking an order-dependent operation when the order is `expr binop x`
// CHECK-LABEL: @omp_atomic_update_ordering
// CHECK-SAME: (ptr %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_update_ordering(%x:!llvm.ptr, %expr: i32) {
  // CHECK: %[[t1:.*]] = shl i32 %[[expr]], %[[x_old:[^ ,]*]]
  // CHECK: store i32 %[[t1]], ptr %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, ptr %[[x_new]]
  // CHECK: cmpxchg ptr %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr {
  ^bb0(%xval: i32):
    %newval = llvm.shl %expr, %xval : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking an order-dependent operation when the order is `x binop expr`
// CHECK-LABEL: @omp_atomic_update_ordering
// CHECK-SAME: (ptr %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_update_ordering(%x:!llvm.ptr, %expr: i32) {
  // CHECK: %[[t1:.*]] = shl i32 %[[x_old:.*]], %[[expr]]
  // CHECK: store i32 %[[t1]], ptr %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, ptr %[[x_new]]
  // CHECK: cmpxchg ptr %[[x]], i32 %[[x_old]], i32 %[[t2]] monotonic
  omp.atomic.update %x : !llvm.ptr {
  ^bb0(%xval: i32):
    %newval = llvm.shl %xval, %expr : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking intrinsic translation.
// CHECK-LABEL: @omp_atomic_update_intrinsic
// CHECK-SAME: (ptr %[[x:.*]], i32 %[[expr:.*]])
llvm.func @omp_atomic_update_intrinsic(%x:!llvm.ptr, %expr: i32) {
  // CHECK: %[[t1:.*]] = call i32 @llvm.smax.i32(i32 %[[x_old:.*]], i32 %[[expr]])
  // CHECK: store i32 %[[t1]], ptr %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, ptr %[[x_new]]
  // CHECK: cmpxchg ptr %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr {
  ^bb0(%xval: i32):
    %newval = "llvm.intr.smax"(%xval, %expr) : (i32, i32) -> i32
    omp.yield(%newval : i32)
  }
  // CHECK: %[[t1:.*]] = call i32 @llvm.umax.i32(i32 %[[x_old:.*]], i32 %[[expr]])
  // CHECK: store i32 %[[t1]], ptr %[[x_new:.*]]
  // CHECK: %[[t2:.*]] = load i32, ptr %[[x_new]]
  // CHECK: cmpxchg ptr %[[x]], i32 %[[x_old]], i32 %[[t2]]
  omp.atomic.update %x : !llvm.ptr {
  ^bb0(%xval: i32):
    %newval = "llvm.intr.umax"(%xval, %expr) : (i32, i32) -> i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_capture_prefix_update
// CHECK-SAME: (ptr %[[x:.*]], ptr %[[v:.*]], i32 %[[expr:.*]], ptr %[[xf:.*]], ptr %[[vf:.*]], float %[[exprf:.*]])
llvm.func @omp_atomic_capture_prefix_update(
  %x: !llvm.ptr, %v: !llvm.ptr, %expr: i32,
  %xf: !llvm.ptr, %vf: !llvm.ptr, %exprf: f32) -> () {
  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = add i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[res:.*]] = atomicrmw sub ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = sub i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.sub %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[res:.*]] = atomicrmw and ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = and i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.and %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[res:.*]] = atomicrmw or ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = or i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.or %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[res:.*]] = atomicrmw xor ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK-NEXT: %[[newval:.*]] = xor i32 %[[res]], %[[expr]]
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.xor %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = mul i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.mul %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = sdiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.sdiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = udiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.udiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = shl i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.shl %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = lshr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.lshr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = ashr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.ashr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[newval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[newval:.*]] = fadd float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], ptr %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK: %{{.*}} = cmpxchg ptr %[[xf]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[newval]], ptr %[[vf]]
  omp.atomic.capture {
    omp.atomic.update %xf : !llvm.ptr {
    ^bb0(%xval: f32):
      %newval = llvm.fadd %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
    omp.atomic.read %vf = %xf : !llvm.ptr, f32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[newval:.*]] = fsub float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], ptr %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK: %{{.*}} = cmpxchg ptr %[[xf]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[newval]], ptr %[[vf]]
  omp.atomic.capture {
    omp.atomic.update %xf : !llvm.ptr {
    ^bb0(%xval: f32):
      %newval = llvm.fsub %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
    omp.atomic.read %vf = %xf : !llvm.ptr, f32
  }

  llvm.return
}

// -----

// CHECK-LABEL: @omp_atomic_capture_postfix_update
// CHECK-SAME: (ptr %[[x:.*]], ptr %[[v:.*]], i32 %[[expr:.*]], ptr %[[xf:.*]], ptr %[[vf:.*]], float %[[exprf:.*]])
llvm.func @omp_atomic_capture_postfix_update(
  %x: !llvm.ptr, %v: !llvm.ptr, %expr: i32,
  %xf: !llvm.ptr, %vf: !llvm.ptr, %exprf: f32) -> () {
  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw sub ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.sub %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw and ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.and %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw or ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.or %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw xor ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.xor %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = mul i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.mul %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = sdiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.sdiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = udiv i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.udiv %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = shl i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.shl %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = lshr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.lshr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = ashr i32 %[[xval]], %[[expr]]
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.ashr %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.smin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.smin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umax.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umax"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK-NEXT: %[[newval:.*]] = call i32 @llvm.umin.i32(i32 %[[xval]], i32 %[[expr]])
  // CHECK-NEXT: store i32 %[[newval]], ptr %{{.*}}
  // CHECK-NEXT: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK-NEXT: %{{.*}} = cmpxchg ptr %[[x]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = "llvm.intr.umin"(%xval, %expr) : (i32, i32) -> i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[xvalf:.*]] = bitcast i32 %[[xval]] to float
  // CHECK: %[[newval:.*]] = fadd float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], ptr %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK: %{{.*}} = cmpxchg ptr %[[xf]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[xvalf]], ptr %[[vf]]
  omp.atomic.capture {
    omp.atomic.read %vf = %xf : !llvm.ptr, f32
    omp.atomic.update %xf : !llvm.ptr {
    ^bb0(%xval: f32):
      %newval = llvm.fadd %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[xvalf:.*]] = bitcast i32 %[[xval]] to float
  // CHECK: %[[newval:.*]] = fsub float %{{.*}}, %[[exprf]]
  // CHECK: store float %[[newval]], ptr %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK: %{{.*}} = cmpxchg ptr %[[xf]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[xvalf]], ptr %[[vf]]
  omp.atomic.capture {
    omp.atomic.read %vf = %xf : !llvm.ptr, f32
    omp.atomic.update %xf : !llvm.ptr {
    ^bb0(%xval: f32):
      %newval = llvm.fsub %xval, %exprf : f32
      omp.yield(%newval : f32)
    }
  }

  llvm.return
}

// -----
// CHECK-LABEL: @omp_atomic_capture_misc
// CHECK-SAME: (ptr %[[x:.*]], ptr %[[v:.*]], i32 %[[expr:.*]], ptr %[[xf:.*]], ptr %[[vf:.*]], float %[[exprf:.*]])
llvm.func @omp_atomic_capture_misc(
  %x: !llvm.ptr, %v: !llvm.ptr, %expr: i32,
  %xf: !llvm.ptr, %vf: !llvm.ptr, %exprf: f32) -> () {
  // CHECK: %[[xval:.*]] = atomicrmw xchg ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[xval]], ptr %[[v]]
  omp.atomic.capture{
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.write %x = %expr : !llvm.ptr, i32
  }

  // CHECK: %[[xval:.*]] = phi i32
  // CHECK: %[[xvalf:.*]] = bitcast i32 %[[xval]] to float
  // CHECK: store float %[[exprf]], ptr %{{.*}}
  // CHECK: %[[newval_:.*]] = load i32, ptr %{{.*}}
  // CHECK: %{{.*}} = cmpxchg ptr %[[xf]], i32 %[[xval]], i32 %[[newval_]] monotonic monotonic
  // CHECK: store float %[[xvalf]], ptr %[[vf]]
  omp.atomic.capture{
    omp.atomic.read %vf = %xf : !llvm.ptr, f32
    omp.atomic.write %xf = %exprf : !llvm.ptr, f32
  }

  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] seq_cst
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] acquire
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture memory_order(acquire) {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] release
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture memory_order(release) {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] monotonic
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }

  // CHECK: %[[res:.*]] = atomicrmw add ptr %[[x]], i32 %[[expr]] acq_rel
  // CHECK: store i32 %[[res]], ptr %[[v]]
  omp.atomic.capture memory_order(acq_rel) {
    omp.atomic.read %v = %x : !llvm.ptr, i32
    omp.atomic.update %x : !llvm.ptr {
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
llvm.func @omp_sections(%arg0 : i32, %arg1 : i32, %arg2 : !llvm.ptr) -> () {

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
      // CHECK:   store i32 %{{.*}}, ptr %{{.*}}, align 4
      // CHECK:   br label %{{.*}}
      llvm.store %add, %arg2 : i32, !llvm.ptr
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
// CHECK-SAME: (i32 %[[x:.*]], i32 %[[y:.*]], ptr %[[zaddr:.*]])
llvm.func @single(%x: i32, %y: i32, %zaddr: !llvm.ptr) {
  // CHECK: %[[a:.*]] = sub i32 %[[x]], %[[y]]
  %a = llvm.sub %x, %y : i32
  // CHECK: store i32 %[[a]], ptr %[[zaddr]]
  llvm.store %a, %zaddr : i32, !llvm.ptr
  // CHECK: call i32 @__kmpc_single
  omp.single {
    // CHECK: %[[z:.*]] = add i32 %[[x]], %[[y]]
    %z = llvm.add %x, %y : i32
    // CHECK: store i32 %[[z]], ptr %[[zaddr]]
    llvm.store %z, %zaddr : i32, !llvm.ptr
    // CHECK: call void @__kmpc_end_single
    // CHECK: call void @__kmpc_barrier
    omp.terminator
  }
  // CHECK: %[[b:.*]] = mul i32 %[[x]], %[[y]]
  %b = llvm.mul %x, %y : i32
  // CHECK: store i32 %[[b]], ptr %[[zaddr]]
  llvm.store %b, %zaddr : i32, !llvm.ptr
  // CHECK: ret void
  llvm.return
}

// -----

// CHECK-LABEL: @single_nowait
// CHECK-SAME: (i32 %[[x:.*]], i32 %[[y:.*]], ptr %[[zaddr:.*]])
llvm.func @single_nowait(%x: i32, %y: i32, %zaddr: !llvm.ptr) {
  // CHECK: %[[a:.*]] = sub i32 %[[x]], %[[y]]
  %a = llvm.sub %x, %y : i32
  // CHECK: store i32 %[[a]], ptr %[[zaddr]]
  llvm.store %a, %zaddr : i32, !llvm.ptr
  // CHECK: call i32 @__kmpc_single
  omp.single nowait {
    // CHECK: %[[z:.*]] = add i32 %[[x]], %[[y]]
    %z = llvm.add %x, %y : i32
    // CHECK: store i32 %[[z]], ptr %[[zaddr]]
    llvm.store %z, %zaddr : i32, !llvm.ptr
    // CHECK: call void @__kmpc_end_single
    // CHECK-NOT: call void @__kmpc_barrier
    omp.terminator
  }
  // CHECK: %[[t:.*]] = mul i32 %[[x]], %[[y]]
  %t = llvm.mul %x, %y : i32
  // CHECK: store i32 %[[t]], ptr %[[zaddr]]
  llvm.store %t, %zaddr : i32, !llvm.ptr
  // CHECK: ret void
  llvm.return
}

// -----

// CHECK: @_QFsubEx = internal global i32 undef
// CHECK: @_QFsubEx.cache = common global ptr null

// CHECK-LABEL: @omp_threadprivate
llvm.func @omp_threadprivate() {
// CHECK:  [[THREAD:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB:[0-9]+]])
// CHECK:  [[TMP1:%.*]] = call ptr @__kmpc_threadprivate_cached(ptr @[[GLOB]], i32 [[THREAD]], ptr @_QFsubEx, i64 4, ptr @_QFsubEx.cache)
// CHECK:  store i32 1, ptr [[TMP1]], align 4
// CHECK:  store i32 3, ptr [[TMP1]], align 4

// CHECK-LABEL: omp.par.region{{.*}}
// CHECK:  [[THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB2:[0-9]+]])
// CHECK:  [[TMP3:%.*]] = call ptr @__kmpc_threadprivate_cached(ptr @[[GLOB2]], i32 [[THREAD2]], ptr @_QFsubEx, i64 4, ptr @_QFsubEx.cache)
// CHECK:  store i32 2, ptr [[TMP3]], align 4

  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.mlir.constant(3 : i32) : i32

  %3 = llvm.mlir.addressof @_QFsubEx : !llvm.ptr
  %4 = omp.threadprivate %3 : !llvm.ptr -> !llvm.ptr

  llvm.store %0, %4 : i32, !llvm.ptr

  omp.parallel  {
    %5 = omp.threadprivate %3 : !llvm.ptr -> !llvm.ptr
    llvm.store %1, %5 : i32, !llvm.ptr
    omp.terminator
  }

  llvm.store %2, %4 : i32, !llvm.ptr
  llvm.return
}

llvm.mlir.global internal @_QFsubEx() : i32

// -----

// CHECK-LABEL: define void @omp_task
// CHECK-SAME: (i32 %[[x:.+]], i32 %[[y:.+]], ptr %[[zaddr:.+]])
llvm.func @omp_task(%x: i32, %y: i32, %zaddr: !llvm.ptr) {
  // CHECK: %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num({{.+}})
  // CHECK: %[[task_data:.+]] = call ptr @__kmpc_omp_task_alloc
  // CHECK-SAME: (ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 1, i64 40,
  // CHECK-SAME:  i64 0, ptr @[[outlined_fn:.+]])
  // CHECK: call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]])
  omp.task {
    %n = llvm.mlir.constant(1 : i64) : i64
    %valaddr = llvm.alloca %n x i32 : (i64) -> !llvm.ptr
    %val = llvm.load %valaddr : !llvm.ptr -> i32
    %double = llvm.add %val, %val : i32
    llvm.store %double, %valaddr : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK: define internal void @[[outlined_fn]](i32 %[[global_tid:[^ ,]+]])
// CHECK: task.alloca{{.*}}:
// CHECK:   br label %[[task_body:[^, ]+]]
// CHECK: [[task_body]]:
// CHECK:   br label %[[task_region:[^, ]+]]
// CHECK: [[task_region]]:
// CHECK:   %[[alloca:.+]] = alloca i32, i64 1
// CHECK:   %[[val:.+]] = load i32, ptr %[[alloca]]
// CHECK:   %[[newval:.+]] = add i32 %[[val]], %[[val]]
// CHECK:   store i32 %[[newval]], ptr %{{[^, ]+}}
// CHECK:   br label %[[exit_stub:[^, ]+]]
// CHECK: [[exit_stub]]:
// CHECK:   ret void

// -----

// CHECK-LABEL: define void @omp_task_attrs()
llvm.func @omp_task_attrs() -> () attributes {
  target_cpu = "x86-64",
  target_features = #llvm.target_features<["+mmx", "+sse"]>
} {
  // CHECK: %[[task_data:.*]] = call {{.*}}@__kmpc_omp_task_alloc{{.*}}@[[outlined_fn:.*]])
  // CHECK: call {{.*}}@__kmpc_omp_task(
  // CHECK-SAME: ptr %[[task_data]]
  omp.task {
    omp.terminator
  }

  llvm.return
}

// CHECK: define {{.*}} @[[outlined_fn]]{{.*}} #[[attrs:[0-9]+]]
// CHECK: attributes #[[attrs]] = {
// CHECK-SAME: "target-cpu"="x86-64"
// CHECK-SAME: "target-features"="+mmx,+sse"

// -----

// CHECK-LABEL: define void @omp_task_with_deps
// CHECK-SAME: (ptr %[[zaddr:.+]])
// CHECK:  %[[dep_arr_addr:.+]] = alloca [1 x %struct.kmp_dep_info], align 8
// CHECK:  %[[dep_arr_addr_0:.+]] = getelementptr inbounds [1 x %struct.kmp_dep_info], ptr %[[dep_arr_addr]], i64 0, i64 0
// CHECK:  %[[dep_arr_addr_0_val:.+]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[dep_arr_addr_0]], i32 0, i32 0
// CHECK:  %[[dep_arr_addr_0_val_int:.+]] = ptrtoint ptr %0 to i64
// CHECK:  store i64 %[[dep_arr_addr_0_val_int]], ptr %[[dep_arr_addr_0_val]], align 4
// CHECK:  %[[dep_arr_addr_0_size:.+]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[dep_arr_addr_0]], i32 0, i32 1
// CHECK:  store i64 8, ptr %[[dep_arr_addr_0_size]], align 4
// CHECK:  %[[dep_arr_addr_0_kind:.+]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[dep_arr_addr_0]], i32 0, i32 2
// CHECK: store i8 1, ptr %[[dep_arr_addr_0_kind]], align 1
llvm.func @omp_task_with_deps(%zaddr: !llvm.ptr) {
  // CHECK: %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num({{.+}})
  // CHECK: %[[task_data:.+]] = call ptr @__kmpc_omp_task_alloc
  // CHECK-SAME: (ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 1, i64 40,
  // CHECK-SAME:  i64 0, ptr @[[outlined_fn:.+]])
  // CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]], {{.*}})
  omp.task depend(taskdependin -> %zaddr : !llvm.ptr) {
    %n = llvm.mlir.constant(1 : i64) : i64
    %valaddr = llvm.alloca %n x i32 : (i64) -> !llvm.ptr
    %val = llvm.load %valaddr : !llvm.ptr -> i32
    %double = llvm.add %val, %val : i32
    llvm.store %double, %valaddr : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK: define internal void @[[outlined_fn]](i32 %[[global_tid:[^ ,]+]])
// CHECK: task.alloca{{.*}}:
// CHECK:   br label %[[task_body:[^, ]+]]
// CHECK: [[task_body]]:
// CHECK:   br label %[[task_region:[^, ]+]]
// CHECK: [[task_region]]:
// CHECK:   %[[alloca:.+]] = alloca i32, i64 1
// CHECK:   %[[val:.+]] = load i32, ptr %[[alloca]]
// CHECK:   %[[newval:.+]] = add i32 %[[val]], %[[val]]
// CHECK:   store i32 %[[newval]], ptr %{{[^, ]+}}
// CHECK:   br label %[[exit_stub:[^, ]+]]
// CHECK: [[exit_stub]]:
// CHECK:   ret void

// -----

// CHECK-LABEL: define void @omp_task
// CHECK-SAME: (i32 %[[x:.+]], i32 %[[y:.+]], ptr %[[zaddr:.+]])
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @omp_task(%x: i32, %y: i32, %zaddr: !llvm.ptr) {
    // CHECK: %[[diff:.+]] = sub i32 %[[x]], %[[y]]
    %diff = llvm.sub %x, %y : i32
    // CHECK: store i32 %[[diff]], ptr %2
    llvm.store %diff, %zaddr : i32, !llvm.ptr
    // CHECK: %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num({{.+}})
    // CHECK: %[[task_data:.+]] = call ptr @__kmpc_omp_task_alloc
    // CHECK-SAME: (ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 1, i64 40, i64 16,
    // CHECK-SAME: ptr @[[outlined_fn:.+]])
    // CHECK: %[[shareds:.+]] = load ptr, ptr %[[task_data]]
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.+}} %[[shareds]], ptr {{.+}}, i64 16, i1 false)
    // CHECK: call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]])
    omp.task {
      %z = llvm.add %x, %y : i32
      llvm.store %z, %zaddr : i32, !llvm.ptr
      omp.terminator
    }
    // CHECK: %[[prod:.+]] = mul i32 %[[x]], %[[y]]
    %b = llvm.mul %x, %y : i32
    // CHECK: store i32 %[[prod]], ptr %[[zaddr]]
    llvm.store %b, %zaddr : i32, !llvm.ptr
    llvm.return
  }
}

// CHECK: define internal void @[[outlined_fn]](i32 %[[global_tid:[^ ,]+]], ptr %[[task_data:.+]])
// CHECK: task.alloca{{.*}}:
// CHECK:   %[[shareds:.+]] = load ptr, ptr %[[task_data]]
// CHECK:   br label %[[task_body:[^, ]+]]
// CHECK: [[task_body]]:
// CHECK:   br label %[[task_region:[^, ]+]]
// CHECK: [[task_region]]:
// CHECK:   %[[sum:.+]] = add i32 %{{.+}}, %{{.+}}
// CHECK:   store i32 %[[sum]], ptr %{{.+}}
// CHECK:   br label %[[exit_stub:[^, ]+]]
// CHECK: [[exit_stub]]:
// CHECK:   ret void

// -----

llvm.func @par_task_(%arg0: !llvm.ptr {fir.bindc_name = "a"}) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  omp.task   {
    omp.parallel   {
      llvm.store %0, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: @par_task_
// CHECK: %[[TASK_ALLOC:.*]] = call ptr @__kmpc_omp_task_alloc({{.*}}ptr @[[task_outlined_fn:.+]])
// CHECK: call i32 @__kmpc_omp_task({{.*}}, ptr %[[TASK_ALLOC]])
// CHECK: define internal void @[[task_outlined_fn]]
// CHECK: %[[ARG_ALLOC:.*]] = alloca { ptr }, align 8
// CHECK: call void ({{.*}}) @__kmpc_fork_call({{.*}}, ptr @[[parallel_outlined_fn:.+]], ptr %[[ARG_ALLOC]])
// CHECK: define internal void @[[parallel_outlined_fn]]
// -----

llvm.func @foo() -> ()

llvm.func @omp_taskgroup(%x: i32, %y: i32, %zaddr: !llvm.ptr) {
  omp.taskgroup {
    llvm.call @foo() : () -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @omp_taskgroup(
// CHECK-SAME:                             i32 %[[x:.+]], i32 %[[y:.+]], ptr %[[zaddr:.+]])
// CHECK:         br label %[[entry:[^,]+]]
// CHECK:       [[entry]]:
// CHECK:         %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK:         call void @__kmpc_taskgroup(ptr @{{.+}}, i32 %[[omp_global_thread_num]])
// CHECK:         br label %[[omp_taskgroup_region:[^,]+]]
// CHECK:       [[omp_taskgroup_region]]:
// CHECK:         call void @foo()
// CHECK:         br label %[[omp_region_cont:[^,]+]]
// CHECK:       [[omp_region_cont]]:
// CHECK:         br label %[[taskgroup_exit:[^,]+]]
// CHECK:       [[taskgroup_exit]]:
// CHECK:         call void @__kmpc_end_taskgroup(ptr @{{.+}}, i32 %[[omp_global_thread_num]])
// CHECK:         ret void

// -----

llvm.func @foo() -> ()
llvm.func @bar(i32, i32, !llvm.ptr) -> ()

llvm.func @omp_taskgroup_task(%x: i32, %y: i32, %zaddr: !llvm.ptr) {
  omp.taskgroup {
    %c1 = llvm.mlir.constant(1) : i32
    %ptr1 = llvm.alloca %c1 x i8 : (i32) -> !llvm.ptr
    omp.task {
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    omp.task {
      llvm.call @bar(%x, %y, %zaddr) : (i32, i32, !llvm.ptr) -> ()
      omp.terminator
    }
    llvm.br ^bb1
  ^bb1:
    llvm.call @foo() : () -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @omp_taskgroup_task(
// CHECK-SAME:                                  i32 %[[x:.+]], i32 %[[y:.+]], ptr %[[zaddr:.+]])
// CHECK:         %[[structArg:.+]] = alloca { i32, i32, ptr }, align 8
// CHECK:         br label %[[entry:[^,]+]]
// CHECK:       [[entry]]:                                            ; preds = %3
// CHECK:         %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK:         call void @__kmpc_taskgroup(ptr @{{.+}}, i32 %[[omp_global_thread_num]])
// CHECK:         br label %[[omp_taskgroup_region:[^,]+]]
// CHECK:       [[omp_taskgroup_region1:.+]]:
// CHECK:         call void @foo()
// CHECK:         br label %[[omp_region_cont:[^,]+]]
// CHECK:       [[omp_taskgroup_region]]:
// CHECK:         %{{.+}} = alloca i8, align 1
// CHECK:         br label %[[codeRepl:[^,]+]]
// CHECK:       [[codeRepl]]:
// CHECK:         %[[omp_global_thread_num_t1:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK:         %[[t1_alloc:.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %[[omp_global_thread_num_t1]], i32 1, i64 40, i64 0, ptr @[[outlined_task_fn:.+]])
// CHECK:         %{{.+}} = call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %[[omp_global_thread_num_t1]], ptr %[[t1_alloc]])
// CHECK:         br label %[[task_exit:[^,]+]]
// CHECK:       [[task_exit]]:
// CHECK:         br label %[[codeRepl9:[^,]+]]
// CHECK:       [[codeRepl9]]:
// CHECK:         %[[gep1:.+]] = getelementptr { i32, i32, ptr }, ptr %[[structArg]], i32 0, i32 0
// CHECK:         store i32 %[[x]], ptr %[[gep1]], align 4
// CHECK:         %[[gep2:.+]] = getelementptr { i32, i32, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK:         store i32 %[[y]], ptr %[[gep2]], align 4
// CHECK:         %[[gep3:.+]] = getelementptr { i32, i32, ptr }, ptr %[[structArg]], i32 0, i32 2
// CHECK:         store ptr %[[zaddr]], ptr %[[gep3]], align 8
// CHECK:         %[[omp_global_thread_num_t2:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK:         %[[t2_alloc:.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %[[omp_global_thread_num_t2]], i32 1, i64 40, i64 16, ptr @[[outlined_task_fn:.+]])
// CHECK:         %[[shareds:.+]] = load ptr, ptr %[[t2_alloc]]
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[shareds]], ptr align 1 %[[structArg]], i64 16, i1 false)
// CHECK:         %{{.+}} = call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %[[omp_global_thread_num_t2]], ptr %[[t2_alloc]])
// CHECK:         br label %[[task_exit3:[^,]+]]
// CHECK:       [[task_exit3]]:
// CHECK:         br label %[[omp_taskgroup_region1]]
// CHECK:       [[omp_region_cont]]:
// CHECK:         br label %[[taskgroup_exit:[^,]+]]
// CHECK:       [[taskgroup_exit]]:
// CHECK:         call void @__kmpc_end_taskgroup(ptr @{{.+}}, i32 %[[omp_global_thread_num]])
// CHECK:         ret void
// CHECK:       }

// -----

// CHECK-LABEL: @omp_opaque_pointers
// CHECK-SAME: (ptr %[[ARG0:.*]], ptr %[[ARG1:.*]], i32 %[[EXPR:.*]])
llvm.func @omp_opaque_pointers(%arg0 : !llvm.ptr, %arg1: !llvm.ptr, %expr: i32) -> () {
  // CHECK: %[[X1:.*]] = load atomic i32, ptr %[[ARG0]] monotonic, align 4
  // CHECK: store i32 %[[X1]], ptr %[[ARG1]], align 4
  omp.atomic.read %arg1 = %arg0 : !llvm.ptr, i32

  // CHECK: %[[RES:.*]] = atomicrmw add ptr %[[ARG1]], i32 %[[EXPR]] acq_rel
  // CHECK: store i32 %[[RES]], ptr %[[ARG0]]
  omp.atomic.capture memory_order(acq_rel) {
    omp.atomic.read %arg0 = %arg1 : !llvm.ptr, i32
    omp.atomic.update %arg1 : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
  }
  llvm.return
}

// -----

// CHECK: @__omp_rtl_debug_kind = weak_odr hidden constant i32 1
// CHECK: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 1
// CHECK: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 1
// CHECK: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 1
// CHECK: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 1
module attributes {omp.flags = #omp.flags<debug_kind = 1, assume_teams_oversubscription = true, 
                                          assume_threads_oversubscription = true, assume_no_thread_state = true, 
                                          assume_no_nested_parallelism = true>} {}
// -----

// CHECK: @__omp_rtl_debug_kind = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
// CHECK: [[META0:![0-9]+]] = !{i32 7, !"openmp-device", i32 50}
module attributes {omp.flags = #omp.flags<>} {}

// -----

// CHECK: @__omp_rtl_debug_kind = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
// CHECK: [[META0:![0-9]+]] = !{i32 7, !"openmp-device", i32 51}
module attributes {omp.flags = #omp.flags<openmp_device_version = 51>} {}

// -----

// CHECK: @__omp_rtl_debug_kind = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
// CHECK: [[META0:![0-9]+]] = !{i32 7, !"openmp-device", i32 50}
// CHECK: [[META0:![0-9]+]] = !{i32 7, !"openmp", i32 50}
module attributes {omp.version = #omp.version<version = 50>, omp.flags = #omp.flags<>} {}

// -----

// CHECK: [[META0:![0-9]+]] = !{i32 7, !"openmp", i32 51}
// CHECK-NOT: [[META0:![0-9]+]] = !{i32 7, !"openmp-device", i32 50}
module attributes {omp.version = #omp.version<version = 51>} {}

// -----
// CHECK: @__omp_rtl_debug_kind = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
module attributes {omp.flags = #omp.flags<debug_kind = 0, assume_teams_oversubscription = false, 
                                          assume_threads_oversubscription = false, assume_no_thread_state = false, 
                                          assume_no_nested_parallelism = false>} {}

// -----

// CHECK: @__omp_rtl_debug_kind = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 1
// CHECK: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
// CHECK: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 1
// CHECK: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {}

// -----

// CHECK-NOT: @__omp_rtl_debug_kind = weak_odr hidden constant i32 0
// CHECK-NOT: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 1
// CHECK-NOT: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
// CHECK-NOT: @__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 1
// CHECK-NOT: @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true,
                                          no_gpu_lib=true>} {}

// -----

module attributes {omp.is_target_device = false} {
  // CHECK: define void @filter_nohost
  llvm.func @filter_nohost() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (to)>
      } {
    llvm.return
  }

  // CHECK: define void @filter_host
  llvm.func @filter_host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    llvm.return
  }
}

// -----

module attributes {omp.is_target_device = false} {
  // CHECK: define void @filter_nohost
  llvm.func @filter_nohost() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>
      } {
    llvm.return
  }

  // CHECK: define void @filter_host
  llvm.func @filter_host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (enter)>
      } {
    llvm.return
  }
}

// -----

module attributes {omp.is_target_device = true} {
  // CHECK: define void @filter_nohost
  llvm.func @filter_nohost() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (to)>
      } {
    llvm.return
  }

  // CHECK-NOT: define void @filter_host
  llvm.func @filter_host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    llvm.return
  }
}

// -----

module attributes {omp.is_target_device = true} {
  // CHECK: define void @filter_nohost
  llvm.func @filter_nohost() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>
      } {
    llvm.return
  }

  // CHECK-NOT: define void @filter_host
  llvm.func @filter_host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (enter)>
      } {
    llvm.return
  }
}

// -----

llvm.func external @foo_before() -> ()
llvm.func external @foo() -> ()
llvm.func external @foo_after() -> ()

llvm.func @omp_task_final(%boolexpr: i1) {
  llvm.call @foo_before() : () -> ()
  omp.task final(%boolexpr) {
    llvm.call @foo() : () -> ()
    omp.terminator
  }
  llvm.call @foo_after() : () -> ()
  llvm.return
}

// CHECK-LABEL: define void @omp_task_final(
// CHECK-SAME:    i1 %[[boolexpr:.+]]) {
// CHECK:         call void @foo_before()
// CHECK:         br label %[[entry:[^,]+]]
// CHECK:       [[entry]]:
// CHECK:         br label %[[codeRepl:[^,]+]]
// CHECK:       [[codeRepl]]:                                         ; preds = %entry
// CHECK:         %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK:         %[[final_flag:.+]] = select i1 %[[boolexpr]], i32 2, i32 0
// CHECK:         %[[task_flags:.+]] = or i32 %[[final_flag]], 1
// CHECK:         %[[task_data:.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 %[[task_flags]], i64 40, i64 0, ptr @[[task_outlined_fn:.+]])
// CHECK:         %{{.+}} = call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]])
// CHECK:         br label %[[task_exit:[^,]+]]
// CHECK:       [[task_exit]]:
// CHECK:         call void @foo_after()
// CHECK:         ret void

// -----

llvm.func external @foo_before() -> ()
llvm.func external @foo() -> ()
llvm.func external @foo_after() -> ()

llvm.func @omp_task_if(%boolexpr: i1) {
  llvm.call @foo_before() : () -> ()
  omp.task if(%boolexpr) {
    llvm.call @foo() : () -> ()
    omp.terminator
  }
  llvm.call @foo_after() : () -> ()
  llvm.return
}

// CHECK-LABEL: define void @omp_task_if(
// CHECK-SAME:    i1 %[[boolexpr:.+]]) {
// CHECK:         call void @foo_before()
// CHECK:         br label %[[entry:[^,]+]]
// CHECK:       [[entry]]:
// CHECK:         br label %[[codeRepl:[^,]+]]
// CHECK:       [[codeRepl]]:
// CHECK:         %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK:         %[[task_data:.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 1, i64 40, i64 0, ptr @[[task_outlined_fn:.+]])
// CHECK:         br i1 %[[boolexpr]], label %[[true_label:[^,]+]], label %[[false_label:[^,]+]]
// CHECK:       [[true_label]]:
// CHECK:         %{{.+}} = call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]])
// CHECK:         br label %[[if_else_exit:[^,]+]]
// CHECK:       [[false_label:[^,]+]]:                                                ; preds = %codeRepl
// CHECK:         call void @__kmpc_omp_task_begin_if0(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]])
// CHECK:         call void @[[task_outlined_fn]](i32 %[[omp_global_thread_num]])
// CHECK:         call void @__kmpc_omp_task_complete_if0(ptr @{{.+}}, i32 %[[omp_global_thread_num]], ptr %[[task_data]])
// CHECK:         br label %[[if_else_exit]]
// CHECK:       [[if_else_exit]]:
// CHECK:         br label %[[task_exit:[^,]+]]
// CHECK:       [[task_exit]]:
// CHECK:         call void @foo_after()
// CHECK:         ret void

// -----

// Check that OpenMP requires flags are registered by a global constructor.
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
// CHECK-SAME: [{ i32, ptr, ptr } { i32 0, ptr @[[REG_FN:.*]], ptr null }]
// CHECK: define {{.*}} @[[REG_FN]]({{.*}})
// CHECK-NOT: }
// CHECK:   call void @__tgt_register_requires(i64 10)
module attributes {omp.requires = #omp<clause_requires reverse_offload|unified_shared_memory>} {}
