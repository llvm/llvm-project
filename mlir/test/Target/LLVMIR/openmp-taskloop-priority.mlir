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
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop priority(%c1_i32 : i32) private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
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

// CHECK: define void @_QPtest() {
// CHECK:   %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK:   %[[VAL_1:.*]] = alloca i32, i64 1, align 4
// CHECK:   %[[VAL_2:.*]] = alloca i32, i64 1, align 4
// CHECK:   store i32 20, ptr %2, align 4
// CHECK:   br label %[[entry:.*]]

// CHECK: entry:                                            ; preds = %0
// CHECK:   br label %[[omp_private_init:.*]]

// CHECK: omp.private.init:                                 ; preds = %[[entry:.*]]
// CHECK:   %[[ctx_ptr:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ i32 }, ptr null, i32 1) to i64))
// CHECK:   %[[VAL_3:.*]] = getelementptr { i32 }, ptr %[[ctx_ptr]], i32 0, i32 0
// CHECK:   br label %[[omp_private_copy:.*]]

// CHECK: omp.private.copy:                                 ; preds = %[[omp_private_init]]
// CHECK:   br label %[[omp_private_copy1:.*]]

// CHECK: omp.private.copy1:                                ; preds = %[[omp_private_copy]]
// CHECK:   %[[VAL_4:.*]] = load i32, ptr %[[VAL_2]], align 4
// CHECK:   store i32 %[[VAL_4]], ptr %[[VAL_3]], align 4
// CHECK:   br label %[[omp_taskloop_start:.*]]

// CHECK: omp.taskloop.start:                               ; preds = %[[omp_private_copy1]]
// CHECK:   br label %[[codeRepl:.*]]

// CHECK: codeRepl:                                         ; preds = %[[omp_taskloop_start]]
// CHECK:   %[[gep_lb_val:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 0
// CHECK:   store i64 1, ptr %[[gep_lb_val]], align 4
// CHECK:   %[[gep_ub_val:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK:   store i64 5, ptr %[[gep_ub_val]], align 4
// CHECK:   %[[gep_step_val:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 2
// CHECK:   store i64 1, ptr %[[gep_step_val]], align 4
// CHECK:   %[[gep_omp_task_context_ptr:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 3
// CHECK:   store ptr %[[ctx_ptr]], ptr %[[gep_omp_task_context_ptr]], align 8
// CHECK:   %[[omp_global_thread_num:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:   call void @__kmpc_taskgroup(ptr @1, i32 %[[omp_global_thread_num]])
// CHECK:   %[[VAL_5:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[omp_global_thread_num]], i32 33, i64 40, i64 32, ptr @_QPtest..omp_par)
// CHECK:   %[[VAL_6:.*]] = load ptr, ptr %[[VAL_5]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_6]], ptr align 1 %[[structArg]], i64 32, i1 false)
// CHECK:   %[[VAL_7:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_6]], i32 0, i32 0
// CHECK:   %[[VAL_8:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_6]], i32 0, i32 1
// CHECK:   %[[VAL_9:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_6]], i32 0, i32 2
// CHECK:   %[[VAL_10:.*]] = load i64, ptr %[[VAL_9]], align 4
// CHECK:   call void @__kmpc_taskloop(ptr @1, i32 %omp_global_thread_num, ptr %[[VAL_5]], i32 1, ptr %[[VAL_7]], ptr %[[VAL_8]], i64 %[[VAL_10]], i32 0, i32 0, i64 0, ptr @omp_taskloop_dup)
// CHECK:   call void @__kmpc_end_taskgroup(ptr @1, i32 %omp_global_thread_num)
// CHECK:   br label %taskloop.exit

// CHECK: taskloop.exit:                                    ; preds = %[[codeRepl]]
// CHECK:   ret void
// CHECK: }

// -----
