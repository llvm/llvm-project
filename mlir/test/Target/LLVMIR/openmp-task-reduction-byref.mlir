// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.declare_reduction @add_reduction_byref_i32 : !llvm.ptr alloc {
   %0 = llvm.mlir.constant(1 : i64) : i64
   %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
   %2 = llvm.mlir.constant(1 : i64) : i64
   omp.yield(%1 : !llvm.ptr)
} init {
 ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
   %0 = llvm.mlir.constant(0 : i32) : i32
   llvm.store %0, %arg1 : i32, !llvm.ptr
   omp.yield(%arg1 : !llvm.ptr)
} combiner {
 ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
   %0 = llvm.load %arg0 : !llvm.ptr -> i32
   %1 = llvm.load %arg1 : !llvm.ptr -> i32
   %2 = llvm.add %0, %1 : i32
   llvm.store %2, %arg0 : i32, !llvm.ptr
   omp.yield(%arg0 : !llvm.ptr)
}
llvm.func @_QPtest_task_reduction() {
   %0 = llvm.mlir.constant(1 : i64) : i64
   %1 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
   %2 = llvm.mlir.constant(1 : i64) : i64
   omp.taskgroup task_reduction(byref @add_reduction_byref_i32 %1 -> %arg0 : !llvm.ptr) {
     omp.terminator
   }
  llvm.return
} 

//CHECK-LABEL: define void @_QPtest_task_reduction() {
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
//CHECK:   br label %omp.region.cont

//CHECK: omp.region.cont:
//CHECK:   br label %taskgroup.exit

//CHECK: taskgroup.exit:
//CHECK:   call void @__kmpc_end_taskgroup(ptr @{{.+}}, i32 %[[TID]])
//CHECK:   ret void
//CHECK: }

//CHECK: define void @red_init(ptr noalias %[[ARG_1:.*]], ptr noalias %[[ARG_2:.*]]) #2 {
//CHECK: entry:
//CHECK: %[[ALLOCA_1:.*]] = alloca ptr, align 8
//CHECK: %[[ALLOCA_2:.*]] = alloca ptr, align 8
//CHECK: store ptr %[[ARG_1]], ptr %[[ALLOCA_1]], align 8
//CHECK: store ptr %[[ARG_2]], ptr %[[ALLOCA_2]], align 8
//CHECK: %[[LOAD:.*]] = load ptr, ptr %[[ALLOCA_1]], align 8
//CHECK: store i32 0, ptr %[[LOAD]], align 4
//CHECK: ret void
//CHECK: }

//CHECK: define void @red_comb(ptr %[[ARG_1:.*]], ptr %[[ARG_2:.*]]) #2 {
//CHECK: entry:
//CHECK: %[[ALLOCA_1:.*]] = alloca ptr, align 8
//CHECK: %[[ALLOCA_2:.*]] = alloca ptr, align 8
//CHECK: store ptr %[[ARG_1]], ptr %[[ALLOCA_1]], align 8
//CHECK: store ptr %[[ARG_2]], ptr %[[ALLOCA_2]], align 8
//CHECK: %[[LOAD_1:.*]] = load ptr, ptr %[[ALLOCA_1]], align 8
//CHECK: %[[LOAD_2:.*]] = load ptr, ptr %[[ALLOCA_2]], align 8
//CHECK: %[[LOAD_1_I32:.*]] = load i32, ptr %[[LOAD_1]], align 4
//CHECK: %[[LOAD_2_I32:.*]] = load i32, ptr %[[LOAD_2]], align 4
//CHECK: %[[ADD:.*]] = add i32 %[[LOAD_1_I32]], %[[LOAD_2_I32]]
//CHECK: store i32 %[[ADD]], ptr %[[LOAD_1]], align 4
//CHECK: ret void
//CHECK: }
