// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that trying to outline an infinite loop doesn't lead to an assertion
// failure.

llvm.func @parallel_infinite_loop() -> () {
  omp.parallel {
    llvm.br ^bb1
  ^bb1:
    llvm.br ^bb1
  }
  llvm.return
}

// CHECK-LABEL: define void @parallel_infinite_loop() {
// CHECK:         %[[VAL_2:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       omp_parallel:
// CHECK:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 0, ptr @parallel_infinite_loop..omp_par)
// CHECK:         unreachable
// CHECK:       omp.region.cont:                                  ; No predecessors!
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp.par.pre_finalize:                             ; preds = %[[VAL_5:.*]]
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       omp.par.exit:                                     ; preds = %[[VAL_4]]
// CHECK:         ret void
// CHECK:       }

// CHECK-LABEL: define internal void @parallel_infinite_loop..omp_par(
// CHECK-SAME:      ptr noalias %[[TID_ADDR:.*]], ptr noalias %[[ZERO_ADDR:.*]])
// CHECK:       omp.par.entry:
// CHECK:         %[[VAL_7:.*]] = alloca i32, align 4
// CHECK:         %[[VAL_8:.*]] = load i32, ptr %[[VAL_9:.*]], align 4
// CHECK:         store i32 %[[VAL_8]], ptr %[[VAL_7]], align 4
// CHECK:         %[[VAL_10:.*]] = load i32, ptr %[[VAL_7]], align 4
// CHECK:         br label %[[VAL_11:.*]]
// CHECK:       omp.region.after_alloca:                          ; preds = %[[VAL_12:.*]]
// CHECK:         br label %[[VAL_13:.*]]
// CHECK:       omp.par.region:                                   ; preds = %[[VAL_11]]
// CHECK:         br label %[[VAL_14:.*]]
// CHECK:       omp.par.region1:                                  ; preds = %[[VAL_13]]
// CHECK:         br label %[[VAL_15:.*]]
// CHECK:       omp.par.region2:                                  ; preds = %[[VAL_15]], %[[VAL_14]]
// CHECK:         br label %[[VAL_15]]
