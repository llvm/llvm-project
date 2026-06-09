// RUN: mlir-translate -mlir-to-llvmir %s 2>&1 | FileCheck %s

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @launch_(%arg0: !llvm.ptr {fir.bindc_name = "a", llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f64 {bindc_name = "n"} : (i64) -> !llvm.ptr
    %2 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%2 : !llvm.ptr, f64)  -> !llvm.ptr {name = ""}
    %4 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(to) capture(ByRef) members(%3 : [0] : !llvm.ptr) -> !llvm.ptr {name = "a"}
    %5 = omp.map.info var_ptr(%1 : !llvm.ptr, f64) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target nowait map_entries(%4 -> %arg1, %5 -> %arg2, %3 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %two_f = llvm.mlir.constant(2.000000e+00 : f64) : f64
      %one_i = llvm.mlir.constant(1 : index) : i64
      %6 = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      %7 = llvm.load %6 : !llvm.ptr -> !llvm.ptr
      %8 = llvm.getelementptr %7[%one_i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %9 = llvm.load %8 : !llvm.ptr -> f64
      %10 = llvm.fmul %9, %two_f {fastmathFlags = #llvm.fastmath<contract>} : f64
      llvm.store %10, %8 : f64, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: %struct.[[TSK_WTH_PRVTS:.*]] = type { %struct.kmp_task_ompbuilder_t, %struct.[[PRVTS:.*]] }
// CHECK: %struct.kmp_task_ompbuilder_t = type { ptr, ptr, i32, ptr, ptr }
// CHECK: %struct.[[PRVTS]] = type { [5 x ptr], [5 x ptr], [5 x i64] }

// CHECK: define void @launch_(ptr captures(none) %0)
// CHECK: %[[STRUCTARG:.*]] = alloca { ptr, ptr }, align 8
// CHECK: %[[BASEPTRS:.*]] = alloca [5 x ptr], align 8
// CHECK: %[[PTRS:.*]] = alloca [5 x ptr], align 8
// CHECK: %[[MAPPERS:.*]] = alloca [5 x ptr], align 8
// CHECK: %[[SIZES:.*]] = alloca [5 x i64], align 4

// CHECK: %[[VAL_20:.*]] = getelementptr inbounds [5 x ptr], ptr %[[BASEPTRS]], i32 0, i32 0
// CHECK: %[[BASEPTRS_GEP:.*]] = getelementptr inbounds [5 x ptr], ptr %[[BASEPTRS]], i32 0, i32 0
// CHECK: %[[PTRS_GEP:.*]] = getelementptr inbounds [5 x ptr], ptr %[[PTRS]], i32 0, i32 0
// CHECK: %[[SIZES_GEP:.*]] = getelementptr inbounds [5 x i64], ptr %[[SIZES]], i32 0, i32 0

// CHECK: %[[GL_THRD_NUM:.*]] = call i32 @__kmpc_global_thread_num
// CHECK: %[[TASK_DESC:.*]] = call ptr @__kmpc_omp_target_task_alloc(ptr @4, i32 {{.*}}, i32 0, i64 160, i64 16, ptr [[TGT_TSK_PRXY_FNC:.*]], i64 -1)
// CHECK: %[[TSK_PTR:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK_DESC]], i32 0, i32 0
// CHECK: %[[SHAREDS:.*]] = getelementptr inbounds nuw %struct.kmp_task_ompbuilder_t, ptr %[[TSK_PTR]], i32 0, i32 0
// CHECK: %[[SHAREDS_PTR:.*]] = load ptr, ptr %[[SHAREDS]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[SHAREDS_PTR]], ptr align 1 %[[STRUCTARG]], i64 16, i1 false)
// CHECK: %[[VAL_50:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK_DESC]], i32 0, i32 1
// CHECK: %[[VAL_51:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[VAL_50]], i32 0, i32 0
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_51]], ptr align 1 %[[BASEPTRS_GEP]], i64 40, i1 false)
// CHECK: %[[VAL_53:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[VAL_50]], i32 0, i32 1
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_53]], ptr align 1 %[[PTRS_GEP]], i64 40, i1 false)
// CHECK: %[[VAL_54:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[VAL_50]], i32 0, i32 2
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[VAL_54]], ptr align 1 %[[SIZES_GEP]], i64 40, i1 false)
// CHECK: %[[VAL_55:.*]] = call i32 @__kmpc_omp_task(ptr @4, i32 %[[GL_THRD_NUM]], ptr %[[TASK_DESC]])

// CHECK: define internal void @[[WORKER:.*]](i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}) {

// CHECK: define internal void [[TGT_TSK_PRXY_FNC]](i32 %[[THREAD_ID_PARAM:.*]], ptr %[[TASK_DESC_PARAM:.*]]) {
// CHECK: %[[PRIVATE_DATA:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK_DESC_PARAM]], i32 0, i32 1
// CHECK: %[[BASEPTRS:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[PRIVATE_DATA]], i32 0, i32 0
// CHECK: %[[PTRS:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[PRIVATE_DATA]], i32 0, i32 1
// CHECK: %[[SIZES:.*]] = getelementptr inbounds nuw %struct.[[PRVTS]], ptr %[[PRIVATE_DATA]], i32 0, i32 2
// CHECK: %[[STRUCTARG:.*]] = alloca { ptr, ptr }, align 8
// CHECK: %[[TASK:.*]] = getelementptr inbounds nuw %struct.[[TSK_WTH_PRVTS]], ptr %[[TASK_DESC_PARAM]], i32 0, i32 0
// CHECK: %[[SHAREDS:.*]] = getelementptr inbounds nuw %struct.kmp_task_ompbuilder_t, ptr %[[TASK]], i32 0, i32 0
// CHECK: %[[SHAREDS_PTR:.*]] = load ptr, ptr %[[SHAREDS]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[STRUCTARG]], ptr align 1 %[[SHAREDS_PTR]], i64 16, i1 false)
// CHECK: call void @[[WORKER]](i32 %[[THREAD_ID_PARAM]], ptr %[[BASEPTRS]], ptr %[[PTRS]], ptr %[[SIZES]], ptr %[[STRUCTARG]])
