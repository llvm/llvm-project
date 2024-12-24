 // RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

 omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }


  llvm.func @_QPtest_inreduction() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "x", pinned} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %4, %3 : i32, !llvm.ptr
    %5 = llvm.load %3 : !llvm.ptr -> i32
    llvm.store %5, %1 : i32, !llvm.ptr
    omp.task in_reduction(@add_reduction_i32 %3 -> %arg0 : !llvm.ptr) {
      %6 = llvm.load %arg0 : !llvm.ptr -> i32
      %7 = llvm.mlir.constant(1 : i32) : i32
      %8 = llvm.add %6, %7 : i32
      llvm.store %8, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }

//CHECK-LABEL: define void @_QPtest_inreduction() {
//CHECK:         %[[STRUCTARG:.*]] = alloca { i32, ptr }, align 8
//CHECK:         %[[VAL1:.*]] = alloca i32, i64 1, align 4
//CHECK:         %[[VAL2:.*]] = alloca i32, i64 1, align 4
//CHECK:         store i32 0, ptr %[[VAL2]], align 4
//CHECK:         %[[VAL3:.*]] = load i32, ptr %[[VAL2]], align 4
//CHECK:         store i32 %[[VAL3]], ptr %[[VAL1]], align 4
//CHECK:         br label %entry

//CHECK: entry:
//CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
//CHECK:   br label %codeRepl

//CHECK: codeRepl:
//CHECK:   %[[TID2:.*]] = getelementptr { i32, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
//CHECK:   store i32 %[[TID]], ptr %[[TID2]], align 4
//CHECK:   %[[VAL4:.*]] = getelementptr { i32, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
//CHECK:   store ptr %[[VAL2]], ptr %[[VAL4]], align 8
//CHECK:   %[[TID3:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
//CHECK:   %[[VAL5:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[TID3]], i32 1, i64 40, i64 16, ptr @_QPtest_inreduction..omp_par)
//CHECK:   %[[VAL6:.*]] = load ptr, ptr %[[VAL5]], align 8
//CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL6]], ptr align 1 %[[STRUCTARG]], i64 16, i1 false)
//CHECK:   %[[VAL7:.*]] = call i32 @__kmpc_omp_task(ptr @1, i32 %[[TID3]], ptr %[[VAL5]])
//CHECK:   br label %task.exit

//CHECK: task.exit:
//CHECK:   ret void
//CHECK: }

//CHECK-LABEL: define internal void @_QPtest_inreduction..omp_par(i32 %{{.*}}, ptr %{{.*}}) {
//CHECK:       task.alloca:
//CHECK:         %[[VAL9:.*]] = load ptr, ptr %{{.*}}, align 8
//CHECK:         %[[TID4:.*]] = getelementptr { i32, ptr }, ptr %[[VAL9]], i32 0, i32 0
//CHECK:         %[[VAL10:.*]] = load i32, ptr %[[TID4]], align 4
//CHECK:         %[[VAL11:.*]] = getelementptr { i32, ptr }, ptr %[[VAL9]], i32 0, i32 1
//CHECK:         %[[VAL12:.*]] = load ptr, ptr %[[VAL11]], align 8
//CHECK:         br label %task.body

//CHECK:       task.body:
//CHECK:         %[[VAL13:.*]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[VAL10]], ptr null, ptr %[[VAL12]])
//CHECK:         br label %omp.task.region

//CHECK:       omp.task.region:
//CHECK:         %[[VAL14:.*]] = load i32, ptr %[[VAL13]], align 4
//CHECK:         %[[VAL15:.*]] = add i32 %[[VAL14]], 1
//CHECK:         store i32 %[[VAL15]], ptr %[[VAL13]], align 4
//CHECK:         br label %omp.region.cont

//CHECK:       omp.region.cont:
//CHECK:         br label %task.exit.exitStub

//CHECK:       task.exit.exitStub:
//CHECK:         ret void
//CHECK: }

// -----

  llvm.func @_QPtest() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(1 : i32) : i32
    llvm.store %3, %1 : i32, !llvm.ptr
    omp.taskgroup task_reduction(@add_reduction_i32 %1 -> %arg0 : !llvm.ptr) {
      omp.task in_reduction(@add_reduction_i32 %1 -> %arg1 : !llvm.ptr) {
        %4 = llvm.load %arg1 : !llvm.ptr -> i32
        %5 = llvm.mlir.constant(1 : i32) : i32
        %6 = llvm.add %4, %5 : i32
        llvm.store %6, %arg1 : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return 
  }

//CHECK-LABEL:  define void @_QPtest() {
//CHECK:          %[[STRUCTARG:.*]] = alloca { i32, ptr, ptr }, align 8
//CHECK:          %[[X_VAL:.*]] = alloca i32, i64 1, align 4
//CHECK:          store i32 1, ptr %[[X_VAL]], align 4
//CHECK:          %[[KMP_TASKRED_ARRAY:.*]] = alloca [1 x %kmp_taskred_input_t], align 8
//CHECK:          br label %entry

//CHECK:        entry:
//CHECK:          %[[OMP_GLOBAL_THREAD_NUM:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
//CHECK:          call void @__kmpc_taskgroup(ptr @1, i32 %[[OMP_GLOBAL_THREAD_NUM]])
//CHECK:          %[[RED_ELEMENT:.*]] = getelementptr [1 x %kmp_taskred_input_t], ptr %[[KMP_TASKRED_ARRAY]], i32 0, i32 0
//CHECK:          %[[REDUCE_SHAR:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 0
//CHECK:          store ptr %[[X_VAL]], ptr %[[REDUCE_SHAR]], align 8
//CHECK:          %[[REDUCE_ORIG:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 1
//CHECK:          store ptr %[[X_VAL]], ptr %[[REDUCE_ORIG]], align 8
//CHECK:          %[[REDUCE_SIZE:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 2
//CHECK:          store i64 4, ptr %[[REDUCE_SIZE]], align 4
//CHECK:          %[[REDUCE_INIT:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 3
//CHECK:          store ptr @red_init, ptr %[[REDUCE_INIT]], align 8
//CHECK:          %[[REDUCE_FINI:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 4
//CHECK:          store ptr null, ptr %[[REDUCE_FINI]], align 8
//CHECK:          %[[REDUCE_COMB:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT:.*]], i32 0, i32 5
//CHECK:          store ptr @red_comb, ptr %[[REDUCE_COMB]], align 8
//CHECK:          %[[FLAGS:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 6
//CHECK:          store i64 0, ptr %[[FLAGS]], align 4
//CHECK:          %[[OMP_GLOBAL_THREAD_NUM1:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
//CHECK:          %[[TASKRED_INIT_CALL:.*]] = call ptr @__kmpc_taskred_init(i32 %[[OMP_GLOBAL_THREAD_NUM1]], i32 1, ptr %[[KMP_TASKRED_ARRAY]])
//CHECK:          br label %omp.taskgroup.region

//CHECK:        omp.taskgroup.region:
//CHECK:          %[[OMP_GLOBAL_THREAD_NUM2:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
//CHECK:          br label %codeRepl

//CHECK:        codeRepl:
//CHECK:          %[[GEP_OMP_GLOBAL_THREAD_NUM2:.*]] = getelementptr { i32, ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
//CHECK:          store i32 %[[OMP_GLOBAL_THREAD_NUM2]], ptr %[[GEP_OMP_GLOBAL_THREAD_NUM2]], align 4
//CHECK:          %[[GEP_:.*]] = getelementptr { i32, ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
//CHECK:          store ptr %[[TASKRED_INIT_CALL]], ptr %[[GEP_]], align 8
//CHECK:          %[[GEP_4:.*]] = getelementptr { i32, ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 2
//CHECK:          store ptr %1, ptr %[[GEP_4]], align 8
//CHECK:          %[[OMP_GLOBAL_THREAD_NUM5:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
//CHECK:          %[[TASK_ALLOC:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[OMP_GLOBAL_THREAD_NUM5]], i32 1, i64 40, i64 24, ptr @_QPtest..omp_par)
//CHECK:          %[[VAL1:.*]] = load ptr, ptr %[[TASK_ALLOC]], align 8
//CHECK:          call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL1]], ptr align 1 %[[STRUCTARG]], i64 24, i1 false)
//CHECK:          %[[VAL2:.*]] = call i32 @__kmpc_omp_task(ptr @1, i32 %[[OMP_GLOBAL_THREAD_NUM5]], ptr %3)
//CHECK:          br label %task.exit

//CHECK:        task.exit:
//CHECK:          br label %omp.region.cont

//CHECK:        omp.region.cont:
//CHECK:          br label %taskgroup.exit

//CHECK:        taskgroup.exit:
//CHECK:          call void @__kmpc_end_taskgroup(ptr @1, i32 %[[OMP_GLOBAL_THREAD_NUM]])
//CHECK:          ret void
//CHECK:        }

//CHECK:        define internal void @_QPtest..omp_par(i32 %global.tid.val, ptr %0) {
//CHECK:        task.alloca:
//CHECK:          %[[VAL3:.*]] = load ptr, ptr %0, align 8
//CHECK:          %[[GEP_OMP_GLOBAL_THREAD_NUM2:.*]] = getelementptr { i32, ptr, ptr }, ptr %[[VAL3]], i32 0, i32 0
//CHECK:          %[[LOADGEP_OMP_GLOBAL_THREAD_NUM2:.*]] = load i32, ptr %[[GEP_OMP_GLOBAL_THREAD_NUM2:.*]], align 4
//CHECK:          %[[GEP_:.*]] = getelementptr { i32, ptr, ptr }, ptr %[[VAL3]], i32 0, i32 1
//CHECK:          %[[LOADGEP_:.*]] = load ptr, ptr %[[GEP_]], align 8
//CHECK:          %[[GEP_1:.*]] = getelementptr { i32, ptr, ptr }, ptr %[[VAL3]], i32 0, i32 2
//CHECK:          %[[LOADGEP_2:.*]] = load ptr, ptr %[[GEP_1]], align 8
//CHECK:          br label %task.body

//CHECK:        task.body:
//CHECK:          %[[GET_TH_DATA:.*]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[LOADGEP_OMP_GLOBAL_THREAD_NUM2]], ptr %[[LOADGEP_]], ptr %[[LOADGEP_2]])
//CHECK:          br label %omp.task.region

//CHECK:        omp.task.region:
//CHECK:          %[[VAL4:.*]] = load i32, ptr %[[GET_TH_DATA]], align 4
//CHECK:          %[[VAL5:.*]] = add i32 %[[VAL4]], 1
//CHECK:          store i32 %[[VAL5]], ptr %[[GET_TH_DATA]], align 4
//CHECK:          br label %omp.region.cont3

//CHECK:        omp.region.cont3:
//CHECK:          br label %task.exit.exitStub

//CHECK:        task.exit.exitStub:
//CHECK:          ret void
//CHECK:        }

// -----
