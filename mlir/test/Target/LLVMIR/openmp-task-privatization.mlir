// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @privatizer : i32

omp.private {type = firstprivate} @firstprivatizer : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @task_privatization_test() {
  %c0 = llvm.mlir.constant(0: i32) : i32
  %c1 = llvm.mlir.constant(1: i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  llvm.store %c1, %1 : i32, !llvm.ptr

  omp.task private(@privatizer %0 -> %arg0, @firstprivatizer %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    %2 = llvm.load %arg1 : !llvm.ptr -> i32
    llvm.store %2, %arg0 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK:       define void @task_privatization_test()
// CHECK:         %[[STRUCT_ARG:.*]] = alloca { ptr }, align 8
// CHECK:         %[[VAL_0:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_1:.*]] = alloca i32, align 4
// CHECK:         store i32 0, ptr %[[VAL_0]], align 4
// CHECK:         store i32 1, ptr %[[VAL_1]], align 4
// CHECK:         br label %entry
// CHECK:       entry:
// CHECK:         br label %omp.private.init
// CHECK:       omp.private.init:
// CHECK:         %[[VAL_5:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ([[STRUCT_KMP_PRIVATES_T:.*]], ptr null, i32 1) to i64))
// CHECK:         %[[VAL_7:.*]] = getelementptr { i32, i32 }, ptr %[[VAL_5]], i32 0, i32 0
// CHECK:         %[[VAL_8:.*]] = getelementptr { i32, i32 }, ptr %[[VAL_5]], i32 0, i32 1
// CHECK:         br label %omp.private.copy1
// CHECK:       omp.private.copy1:
// CHECK:         %[[VAL_10:.*]] = load i32, ptr %[[VAL_1]], align 4
// CHECK:         store i32 %[[VAL_10]], ptr %[[VAL_8]], align 4
// CHECK:         br label %omp.private.copy
// CHECK:       omp.private.copy:
// CHECK:         br label %omp.task.start
// CHECK:       omp.task.start:
// CHECK:         br label %codeRepl
// CHECK:       codeRepl:
// CHECK:         %[[GEP_OMP_TASK_CONTEXT_PTR:.*]] = getelementptr { ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 0
// CHECK:         store ptr %[[VAL_5]], ptr %[[GEP_OMP_TASK_CONTEXT_PTR]], align 8
// CHECK:         %[[VAL_14:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_15:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[VAL_14]], i32 1, i64 40, i64 8, ptr @task_privatization_test..omp_par)
// CHECK:         %[[ALLOCATED_TASK_STRUCT:.*]] = load ptr, ptr %[[VAL_15]], align 8
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[ALLOCATED_TASK_STRUCT]], ptr align 1 %[[STRUCT_ARG]], i64 8, i1 false)
// CHECK:         %[[VAL_16:.*]] = call i32 @__kmpc_omp_task(ptr @1, i32 %[[VAL_14]], ptr %[[VAL_15]])
// CHECK:         br label %[[VAL_17:.*]]
// CHECK:       task.exit:
// CHECK:         ret void

// CHECK-LABEL: define internal void @task_privatization_test..omp_par(
// CHECK-SAME:      i32 %[[GLOBAL_TID_VAL:.*]], ptr %[[OMP_TASK_CONTEXT_PTR_PTR_PTR_PTR:.*]])
// CHECK:       task.alloca:
// CHECK:         %[[OMP_TASK_CONEXT_PTR_PTR_PTR:.*]] = load ptr, ptr %[[OMP_TASK_CONTEXT_PTR_PTR_PTR_PTR]], align 8
// CHECK:         %[[OMP_TASK_CONTEXT_PTR_PTR:.*]] = getelementptr { ptr }, ptr %[[OMP_TASK_CONTEXT_PTR_PTR_PTR:.*]], i32 0, i32 0
// CHECK:         %[[OMP_TASK_CONTEXT_PTR:.*]] = load ptr, ptr %[[OMP_TASK_CONTEXT_PTR_PTR:.*]], align 8
// CHECK:         br label %[[VAL_18:.*]]
// CHECK:       task.body:                                        ; preds = %[[VAL_19:.*]]
// CHECK:         %[[VAL_20:.*]] = getelementptr { i32, i32 }, ptr %[[OMP_TASK_CONTEXT_PTR]], i32 0, i32 0
// CHECK:         %[[VAL_22:.*]] = getelementptr { i32, i32 }, ptr %[[OMP_TASK_CONTEXT_PTR]], i32 0, i32 1
// CHECK:         br label %[[VAL_23:.*]]
// CHECK:       omp.task.region:                                  ; preds = %[[VAL_18]]
// CHECK:         %[[VAL_24:.*]] = load i32, ptr %[[VAL_22]], align 4
// CHECK:         store i32 %[[VAL_24]], ptr %[[VAL_20]], align 4
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_23]]
// CHECK:         tail call void @free(ptr %[[OMP_TASK_CONTEXT_PTR]])
// CHECK:         br label %[[VAL_26:.*]]
// CHECK:       task.exit.exitStub:                               ; preds = %[[VAL_25]]
// CHECK:         ret void

