// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

llvm.func @cancel_parallel() {
  omp.parallel {
    omp.cancel cancellation_construct_type(parallel)
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancel_parallel..omp_par
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_5:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_6:.*]] = load i32, ptr %[[VAL_7:.*]], align 4
// CHECK:         store i32 %[[VAL_6]], ptr %[[VAL_5]], align 4
// CHECK:         %[[VAL_8:.*]] = load i32, ptr %[[VAL_5]], align 4
// CHECK:         br label %[[VAL_9:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_10:.*]]
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_9]]
// CHECK:         br label %[[VAL_12:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_11]]
// CHECK:         %[[VAL_13:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_14:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_13]], i32 1)
// CHECK:         %[[VAL_15:.*]] = icmp eq i32 %[[VAL_14]], 0
// CHECK:         br i1 %[[VAL_15]], label %[[VAL_16:.*]], label %[[VAL_17:.*]]
// CHECK:       omp.par.region1.cncl:                             ; preds = %[[VAL_12]]
// CHECK:         %[[VAL_18:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_19:.*]] = call i32 @__kmpc_cancel_barrier(ptr @2, i32 %[[VAL_18]])
// CHECK:         br label %[[VAL_20:.*]]
// CHECK:       omp.par.region1.split:                            ; preds = %[[VAL_12]]
// CHECK:         br label %[[VAL_21:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_16]]
// CHECK:         br label %[[VAL_22:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_21]]
// CHECK:         br label %[[VAL_20]]
// CHECK:       omp.par.exit.exitStub:                            ; preds = %[[VAL_22]], %[[VAL_17]]
// CHECK:         ret void

llvm.func @cancel_parallel_if(%arg0 : i1) {
  omp.parallel {
    omp.cancel cancellation_construct_type(parallel) if(%arg0)
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancel_parallel_if..omp_par
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_9:.*]] = getelementptr { ptr }, ptr %[[VAL_10:.*]], i32 0, i32 0
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_9]], align 8
// CHECK:         %[[VAL_12:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_13:.*]] = load i32, ptr %[[VAL_14:.*]], align 4
// CHECK:         store i32 %[[VAL_13]], ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_15:.*]] = load i32, ptr %[[VAL_12]], align 4
// CHECK:         %[[VAL_16:.*]] = load i1, ptr %[[VAL_11]], align 1
// CHECK:         br label %[[VAL_17:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_18:.*]]
// CHECK:         br label %[[VAL_19:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_17]]
// CHECK:         br label %[[VAL_20:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_19]]
// CHECK:         br i1 %[[VAL_16]], label %[[VAL_21:.*]], label %[[VAL_22:.*]]
// CHECK:       3:                                                ; preds = %[[VAL_20]]
// CHECK:         br label %[[VAL_23:.*]]
// CHECK:       4:                                                ; preds = %[[VAL_22]], %[[VAL_24:.*]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_23]]
// CHECK:         br label %[[VAL_26:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_27:.*]]
// CHECK:       5:                                                ; preds = %[[VAL_20]]
// CHECK:         %[[VAL_28:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_29:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_28]], i32 1)
// CHECK:         %[[VAL_30:.*]] = icmp eq i32 %[[VAL_29]], 0
// CHECK:         br i1 %[[VAL_30]], label %[[VAL_24]], label %[[VAL_31:.*]]
// CHECK:       .cncl:                                            ; preds = %[[VAL_21]]
// CHECK:         %[[VAL_32:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_33:.*]] = call i32 @__kmpc_cancel_barrier(ptr @2, i32 %[[VAL_32]])
// CHECK:         br label %[[VAL_27]]
// CHECK:       .split:                                           ; preds = %[[VAL_21]]
// CHECK:         br label %[[VAL_23]]
// CHECK:       omp.par.exit.exitStub:                            ; preds = %[[VAL_31]], %[[VAL_26]]
// CHECK:         ret void
