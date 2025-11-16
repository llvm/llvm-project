// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @_QFtestEi_private_i32 : i32

omp.private {type = firstprivate} @_QFtestEa_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}


llvm.func @_QPtest() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.mlir.constant(5 : i32) : i32
  %9 = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2) : i32 = (%7) to (%8) inclusive step (%9) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK:  %struct.kmp_task_info = type { ptr, ptr, i32, ptr, ptr, i64, i64, i64 }

// CHECK-LABEL:  define void @_QPtest() {
// CHECK:           %[[STRUCTARG:.*]] = alloca { ptr }, align 8
// CHECK:           %[[VAL1:.*]] = alloca i32, i64 1, align 4
// CHECK:           %[[VAL_X:.*]] = alloca i32, i64 1, align 4
// CHECK:           store i32 20, ptr %[[VAL_X]], align 4
// CHECK:           br label %entry

// CHECK:         entry:
// CHECK:           br label %omp.private.init

// CHECK:         omp.private.init:                                 ; preds = %entry
// CHECK:           %[[OMP_TASK_CONTEXT_PTR:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ i32 }, ptr null, i32 1) to i64))
// CHECK:           %[[PRIV_GEP:.*]] = getelementptr { i32 }, ptr %[[OMP_TASK_CONTEXT_PTR]], i32 0, i32 0
// CHECK:           br label %omp.private.copy

// CHECK:         omp.private.copy:
// CHECK:           br label %omp.private.copy1

// CHECK:         omp.private.copy1:
// CHECK:           %[[LOAD_X:.*]] = load i32, ptr %[[VAL_X]], align 4
// CHECK:           store i32 %[[LOAD_X]], ptr %[[PRIV_GEP]], align 4
// CHECK:           br label %omp.taskloop.start

// CHECK:         omp.taskloop.start:
// CHECK:           br label %codeRepl

// CHECK:         codeRepl:
// CHECK:           %[[GEP_OMP_TASK_CONTEXT_PTR:.*]] = getelementptr { ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// CHECK:           store ptr %[[OMP_TASK_CONTEXT_PTR]], ptr %[[GEP_OMP_TASK_CONTEXT_PTR]], align 8
// CHECK:           %[[GTID:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:           call void @__kmpc_taskgroup(ptr @1, i32 %[[GTID]])
// CHECK:           %[[TASK_PTR:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[GTID]], i32 1, i64 64, i64 8, ptr @_QPtest..omp_par)
// CHECK:           %[[LB_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_task_info, ptr %[[TASK_PTR]], i32 0, i32 5
// CHECK:           store i64 1, ptr %[[LB_GEP]], align 4
// CHECK:           %[[UB_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_task_info, ptr %[[TASK_PTR]], i32 0, i32 6
// CHECK:           store i64 5, ptr %[[UB_GEP]], align 4
// CHECK:           %[[STEP_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_task_info, ptr %[[TASK_PTR]], i32 0, i32 7
// CHECK:           store i64 1, ptr %[[STEP_GEP]], align 4
// CHECK:           %[[LOAD_STEP:.*]] = load i64, ptr %[[STEP_GEP]], align 4
// CHECK:           %10 = load ptr, ptr %[[TASK_PTR]], align 8
// CHECK:           call void @llvm.memcpy.p0.p0.i64(ptr align 1 %10, ptr align 1 %[[STRUCTARG]], i64 8, i1 false)
// CHECK:           call void @__kmpc_taskloop(ptr @1, i32 %[[GTID]], ptr %[[TASK_PTR]], i32 1, ptr %[[LB_GEP]], ptr %[[UB_GEP]], i64 %[[LOAD_STEP]], i32 1, i32 0, i64 0, ptr null)
// CHECK:           call void @__kmpc_end_taskgroup(ptr @1, i32 %[[GTID]])
// CHECK:           br label %taskloop.exit

// CHECK:           taskloop.exit:
// CHECK:             tail call void @free(ptr %[[OMP_TASK_CONTEXT_PTR]])
// CHECK:             ret void
// CHECK:           }

// CHECK-LABEL:   define internal void @_QPtest..omp_par
// CHECK-SAME:       i32 %[[GLOBAL_TID:.*]], ptr %[[TASK_PTR1:.*]]) {
// CHECK:           taskloop.alloca:
// CHECK:           %[[LOAD_TASK_PTR:.*]] = load ptr, ptr %[[TASK_PTR1]], align 8
// CHECK:           %[[GEP_LB:.*]] = getelementptr inbounds nuw %struct.kmp_task_info, ptr %[[TASK_PTR1]], i32 0, i32 5
// CHECK:           %[[LOAD_LB64:.*]] = load i64, ptr %[[GEP_LB]], align 4
// CHECK:           %[[LB:.*]] = trunc i64 %[[LOAD_LB64]] to i32
// CHECK:           %[[GEP_UB:.*]] = getelementptr inbounds nuw %struct.kmp_task_info, ptr %[[TASK_PTR1]], i32 0, i32 6
// CHECK:           %[[LOAD_UB64:.*]] = load i64, ptr %[[GEP_UB]], align 4
// CHECK:           %[[UB:.*]] = trunc i64 %[[LOAD_UB64]] to i32
// CHECK:           %[[GEP_OMP_TASK_CONTEXT_PTR:.*]] = getelementptr { ptr }, ptr %[[LOAD_TASK_PTR]], i32 0, i32 0
// CHECK:           %[[LOADGEP_OMP_TASK_CONTEXT_PTR:.*]] = load ptr, ptr %[[GEP_OMP_TASK_CONTEXT_PTR]], align 8, !align !1
// CHECK:           %[[OMP_PRIVATE_ALLOC:.*]] = alloca i32, align 4
// CHECK:           br label %taskloop.body

// CHECK:           taskloop.body:
// CHECK:             %[[LOAD_X:.*]] = getelementptr { i32 }, ptr %[[LOADGEP_OMP_TASK_CONTEXT_PTR]], i32 0, i32 0
// CHECK:             br label %omp.taskloop.region

// CHECK:           omp.taskloop.region:
// CHECK:             br label %omp_loop.preheader

// CHECK:           omp_loop.preheader:
// CHECK:             %[[VAL2:.*]] = sub i32 %[[UB]], %[[LB]]
// CHECK:             %[[TRIP_CNT:.*]] = add i32 %[[VAL2]], 1
// CHECK:             br label %omp_loop.header

// CHECK:           omp_loop.header:
// CHECK:             %[[OMP_LOOP_IV:.*]] = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// CHECK:             br label %omp_loop.cond

// CHECK:           omp_loop.cond:
// CHECK:             %[[OMP_LOOP_CMP:.*]] = icmp ult i32 %[[OMP_LOOP_IV]], %[[TRIP_CNT]]
// CHECK:             br i1 %[[OMP_LOOP_CMP]], label %omp_loop.body, label %omp_loop.exit

// CHECK:           omp_loop.exit:
// CHECK:             br label %omp_loop.after

// CHECK:           omp_loop.after:
// CHECK:             br label %omp.region.cont

// CHECK:           omp.region.cont:
// CHECK:             %[[IS_ALLOCATED:.*]] = icmp ne ptr %[[LOADGEP_OMP_TASK_CONTEXT_PTR]], null
// CHECK:             br label %taskloop.exit.exitStub

// CHECK:           omp_loop.body:
// CHECK:             %[[VAL3:.*]] = mul i32 %[[OMP_LOOP_IV]], 1
// CHECK:             %[[VAL5:.*]] = add i32 %[[VAL3]], %[[LB]]
// CHECK:             br label %omp.loop_nest.region

// CHECK:           omp.loop_nest.region:
// CHECK:             store i32 %[[VAL5]], ptr %[[OMP_PRIVATE_ALLOC]], align 4
// CHECK:             %[[VAL6:.*]] = load i32, ptr %[[LOAD_X]], align 4
// CHECK:             %[[RES:.*]] = add i32 %[[VAL6]], 1
// CHECK:             store i32 %[[RES]], ptr %[[LOAD_X]], align 4
// CHECK:             br label %omp.region.cont2

// CHECK:           omp.region.cont2:
// CHECK:             br label %omp_loop.inc

// CHECK:           omp_loop.inc:
// CHECK:             %omp_loop.next = add nuw i32 %[[OMP_LOOP_IV]], 1
// CHECK:             br label %omp_loop.header

// CHECK:           taskloop.exit.exitStub:
// CHECK:             ret void
// CHECK:           }