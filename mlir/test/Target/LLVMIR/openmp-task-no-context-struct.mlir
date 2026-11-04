// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that we don't allocate a task context structure when none of the private
// vars need it.

omp.private {type = private} @_QFtestEp_private_i32 : i32
llvm.func @_QPtest() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "p"} : (i64) -> !llvm.ptr
  omp.task private(@_QFtestEp_private_i32 %1 -> %arg0 : !llvm.ptr) {
    llvm.call @_QPdo_something(%arg0) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}
llvm.func @_QPdo_something(!llvm.ptr) attributes {sym_visibility = "private"}

// CHECK-LABEL: define void @_QPtest()
// CHECK:         %[[VAL_0:.*]] = alloca i32, i64 1, align 4
// CHECK:         br label %[[VAL_1:.*]]
// CHECK:       entry:                                            ; preds = %[[VAL_2:.*]]
// CHECK:         br label %[[VAL_3:.*]]
// CHECK:       omp.private.init:                                 ; preds = %[[VAL_1]]
// CHECK-NOT:     @malloc
// CHECK:         br label %[[VAL_4:.*]]
// CHECK:       omp.private.copy:                                 ; preds = %[[VAL_3]]
// CHECK:         br label %[[VAL_5:.*]]
// CHECK:       omp.task.start:                                   ; preds = %[[VAL_4]]
// CHECK:         br label %[[VAL_6:.*]]
// CHECK:       codeRepl:                                         ; preds = %[[VAL_5]]
// CHECK:         %[[VAL_7:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_8:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[VAL_7]], i32 1, i64 40, i64 0, ptr @_QPtest..omp_par)
// CHECK:         %[[VAL_9:.*]] = call i32 @__kmpc_omp_task(ptr @1, i32 %[[VAL_7]], ptr %[[VAL_8]])
// CHECK:         br label %[[VAL_10:.*]]
// CHECK:       task.exit:                                        ; preds = %[[VAL_6]]
// CHECK:         ret void

// CHECK-LABEL: define internal void @_QPtest..omp_par
// CHECK:       task.alloca:
// CHECK:         %[[VAL_11:.*]] = alloca i32, align 4
// CHECK:         br label %[[VAL_12:.*]]
// CHECK:       task.body:                                        ; preds = %[[VAL_13:.*]]
// CHECK:         br label %[[VAL_14:.*]]
// CHECK:       omp.task.region:                                  ; preds = %[[VAL_12]]
// CHECK:         call void @_QPdo_something(ptr %[[VAL_11]])
// CHECK:         br label %[[VAL_15:.*]]
// CHECK:       omp.region.cont:                                  ; preds = %[[VAL_14]]
// CHECK-NOT:     @free
