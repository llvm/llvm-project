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
llvm.func @_QPtest_task_reduciton() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  omp.taskgroup task_reduction(@add_reduction_i32 %1 -> %arg0 : !llvm.ptr) {
      %2 = llvm.load %1 : !llvm.ptr -> i32
      %3 = llvm.mlir.constant(1 : i32) : i32
      %4 = llvm.add %2, %3 : i32
      llvm.store %4, %1 : i32, !llvm.ptr
      omp.terminator
  }
  llvm.return
}

//CHECK-LABEL: define void @_QPtest_task_reduciton() {
//CHECK:   %[[VAL1:.*]] = alloca i32, i64 1, align 4
//CHECK:   %[[RED_ARRY:.*]] = alloca [1 x %kmp_taskred_input_t], align 8
//CHECK:   br label %entry

//CHECK: entry:
//CHECK:   %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
//CHECK:   call void @__kmpc_taskgroup(ptr @1, i32 %[[TID]])
//CHECK:   %[[RED_ELEMENT:.*]] = getelementptr [1 x %kmp_taskred_input_t], ptr %[[RED_ARRY]], i32 0, i32 0
//CHECK:   %[[RED_SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 0
//CHECK:   store ptr %[[VAL1]], ptr %[[RED_SHARED]], align 8
//CHECK:   %[[RED_ORIG:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 1
//CHECK:   store ptr %[[VAL1]], ptr %[[RED_ORIG]], align 8
//CHECK:   %[[RED_SIZE:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 2
//CHECK:   store i64 4, ptr %[[RED_SIZE]], align 4
//CHECK:   %[[RED_INIT:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 3
//CHECK:   store ptr @red_init, ptr %[[RED_INIT]], align 8
//CHECK:   %[[RED_FINI:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 4
//CHECK:   store ptr null, ptr %[[RED_FINI]], align 8
//CHECK:   %[[RED_COMB:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 5
//CHECK:   store ptr @red_comb, ptr %[[RED_COMB]], align 8
//CHECK:   %[[FLAGS:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[RED_ELEMENT]], i32 0, i32 6
//CHECK:   store i64 0, ptr %[[FLAGS]], align 4
//CHECK:   %[[TID1:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
//CHECK:   %2 = call ptr @__kmpc_taskred_init(i32 %[[TID1]], i32 1, ptr %[[RED_ARRY]])
//CHECK:   br label %omp.taskgroup.region

//CHECK: omp.taskgroup.region:
//CHECK:   %[[VAL3:.*]] = load i32, ptr %[[VAL1]], align 4
//CHECK:   %4 = add i32 %[[VAL3]], 1
//CHECK:   store i32 %4, ptr %[[VAL1]], align 4
//CHECK:   br label %omp.region.cont

//CHECK: omp.region.cont:
//CHECK:   br label %taskgroup.exit

//CHECK: taskgroup.exit:
//CHECK:   call void @__kmpc_end_taskgroup(ptr @{{.+}}, i32 %[[TID]])
//CHECK:   ret void
//CHECK: }

//CHECK-LABEL: define ptr @red_init(ptr noalias %0, ptr noalias %1) #2 {
//CHECK: entry:
//CHECK:   store i32 0, ptr %0, align 4
//CHECK:   ret ptr %0
//CHECK: }

//CHECK-LABEL: define ptr @red_comb(ptr %0, ptr %1) #2 {
//CHECK: entry:
//CHECK:   %[[LD0:.*]] = load i32, ptr %0, align 4
//CHECK:   %[[LD1:.*]] = load i32, ptr %1, align 4
//CHECK:   %[[RES:.*]] = add i32 %[[LD0]], %[[LD1]]
//CHECK:   store i32 %[[RES]], ptr %0, align 4
//CHECK:   ret ptr %0
//CHECK: }
