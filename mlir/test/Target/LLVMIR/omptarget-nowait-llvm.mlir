// RUN: mlir-translate -mlir-to-llvmir %s 2>&1 | FileCheck %s

// Set a dummy target triple to enable target region outlining.
module attributes {omp.target_triples = ["dummy-target-triple"]} {
  llvm.func @_QPfoo() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr
    omp.target nowait map_entries(%2 -> %arg0 : !llvm.ptr) {
      %3 = llvm.mlir.constant(2 : i32) : i32
      llvm.store %3, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: %struct.[[TSK_WTH_PRVTS:.*]] = type { %struct.kmp_task_ompbuilder_t, %struct.[[PRVTS:.*]] }
// CHECK: %struct.kmp_task_ompbuilder_t = type { ptr, ptr, i32, ptr, ptr }
// CHECK: %struct.[[PRVTS]] = type { [1 x ptr], [1 x ptr] }

// CHECK: define void @_QPfoo() {
// CHECK: %[[STRUCTARG:.*]] = alloca { ptr }, align 8
// CHECK: %[[BASEPTRS:.*]] = alloca [1 x ptr], align 8
// CHECK: %[[PTRS:.*]] = alloca [1 x ptr], align 8
// CHECK: %[[MAPPERS:.*]] = alloca [1 x ptr], align 8

// CHECK: getelementptr inbounds [1 x ptr], ptr %[[BASEPTRS]], i32 0, i32 0
// CHECK: getelementptr inbounds [1 x ptr], ptr %[[PTRS]], i32 0, i32 0
// CHECK: %[[BASEPTRS_GEP:.*]] = getelementptr inbounds [1 x ptr], ptr %[[BASEPTRS]], i32 0, i32 0
// CHECK: %[[PTRS_GEP:.*]] = getelementptr inbounds [1 x ptr], ptr %[[PTRS]], i32 0, i32 0


// CHECK: %[[TASK:.*]] = call ptr @__kmpc_omp_target_task_alloc
// CHECK-SAME: (ptr @{{.*}}, i32 %{{.*}}, i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, ptr
// CHECK-SAME: @[[TASK_PROXY_FUNC:.*]], i64 {{.*}})
// CHECK: %[[TSK_PTR:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK]], i32 0, i32 0
// CHECK: %[[SHAREDS:.*]] = getelementptr inbounds nuw %struct.kmp_task_ompbuilder_t, ptr %[[TSK_PTR]], i32 0, i32 0
// CHECK: %[[SHAREDS_PTR:.*]] = load ptr, ptr %[[SHAREDS]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[SHAREDS_PTR]], ptr align 1 %[[STRUCTARG]], i64 8, i1 false)
// CHECK: %[[VAL_50:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK]], i32 0, i32 1
// CHECK: %[[VAL_51:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[VAL_50]], i32 0, i32 0
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_51]], ptr align 1 %[[BASEPTRS_GEP]], i64 8, i1 false)
// CHECK: %[[VAL_53:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[VAL_50]], i32 0, i32 1
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_53]], ptr align 1 %[[PTRS_GEP]], i64 8, i1 false)
// CHECK: call i32 @__kmpc_omp_task(ptr {{.*}}, i32 %{{.*}}, ptr %[[TASK]])
// CHECK: }

// CHECK: define internal void @[[WORKER:.*]](i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}) {

// CHECK: define internal void @[[TASK_PROXY_FUNC]](i32 %[[THREAD_ID_PARAM:.*]], ptr %[[TASK_DESC_PARAM:.*]]) {
// CHECK: %[[PRIVATE_DATA:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK_DESC_PARAM]], i32 0, i32 1
// CHECK: %[[BASEPTRS:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[PRIVATE_DATA]], i32 0, i32 0
// CHECK: %[[PTRS:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[PRIVATE_DATA]], i32 0, i32 1
// CHECK: %[[STRUCTARG:.*]] = alloca { ptr }, align 8
// CHECK: %[[TASK:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK_DESC_PARAM]], i32 0, i32 0
// CHECK: %[[SHAREDS:.*]] = getelementptr inbounds nuw %struct.kmp_task_ompbuilder_t, ptr %[[TASK]], i32 0, i32 0
// CHECK: %[[SHAREDS_PTR:.*]] = load ptr, ptr %[[SHAREDS]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[STRUCTARG]], ptr align 1 %[[SHAREDS_PTR]], i64 8, i1 false)
// CHECK:   call void @[[WORKER]](i32 %{{.*}}, ptr %{{.*}})
