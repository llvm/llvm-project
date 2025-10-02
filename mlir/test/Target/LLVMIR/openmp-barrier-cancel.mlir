// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// Regression test for a compiler crash. Ensure that the insertion point is set
// correctly after the barrier's cancel check

llvm.func @test() {
  omp.parallel {
    omp.cancel cancellation_construct_type(parallel)
    omp.barrier
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @test..omp_par
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_4:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_5:.*]] = load i32, ptr %[[VAL_6:.*]], align 4
// CHECK:         store i32 %[[VAL_5]], ptr %[[VAL_4]], align 4
// CHECK:         %[[VAL_7:.*]] = load i32, ptr %[[VAL_4]], align 4
// CHECK:         br label %[[VAL_8:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_9:.*]]
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_8]]
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_10]]
// CHECK:         %[[VAL_12:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_13:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_12]], i32 1)
// CHECK:         %[[VAL_14:.*]] = icmp eq i32 %[[VAL_13]], 0
// CHECK:         br i1 %[[VAL_14]], label %[[VAL_15:.*]], label %[[VAL_16:.*]]
// CHECK:       omp.par.region1.cncl:                             ; preds = %[[VAL_11]]
// CHECK:         %[[VAL_17:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_18:.*]] = call i32 @__kmpc_cancel_barrier(ptr @2, i32 %[[VAL_17]])
// CHECK:         br label %[[VAL_19:.*]]
// CHECK:       omp.par.region1.split:                            ; preds = %[[VAL_11]]
// CHECK:         %[[VAL_20:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_21:.*]] = call i32 @__kmpc_cancel_barrier(ptr @3, i32 %[[VAL_20]])
// CHECK:         %[[VAL_22:.*]] = icmp eq i32 %[[VAL_21]], 0
// CHECK:         br i1 %[[VAL_22]], label %[[VAL_23:.*]], label %[[VAL_24:.*]]
// CHECK:       omp.par.region1.split.cncl:                       ; preds = %[[VAL_15]]
// CHECK:         br label %[[VAL_19]]
// CHECK:       omp.par.region1.split.cont:                       ; preds = %[[VAL_15]]
// CHECK:         br label %[[VAL_25:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_23]]
// CHECK:         br label %[[VAL_26:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_25]]
// CHECK:         br label %[[VAL_19]]
// CHECK:       omp.par.exit.exitStub:                            ; preds = %[[VAL_26]], %[[VAL_24]], %[[VAL_16]]
// CHECK:         ret void

