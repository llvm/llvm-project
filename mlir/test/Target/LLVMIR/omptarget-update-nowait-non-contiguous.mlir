// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// This translates host IR and does not require an AMDGPU backend.
// Deferred updates must own their non-contiguous descriptors as well as their
// base and begin pointer arrays. Different descriptor ranks and intervening
// contiguous entries check that the task rewrites the correct begin pointers.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @deferred_update(%first: !llvm.ptr, %array: !llvm.ptr, %middle: !llvm.ptr, %strided: !llvm.ptr, %last: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %inner = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c1 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %outer = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c4 : i64) stride(%c3 : i64) start_idx(%c1 : i64)
    %strided_bound = omp.map.bounds lower_bound(%c1 : i64) upper_bound(%c3 : i64) stride(%c16 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %map_first = omp.map.info var_ptr(%first : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr
    %map_array = omp.map.info var_ptr(%array : !llvm.ptr, !llvm.array<5 x array<3 x i32>>) map_clauses(to) capture(ByRef) bounds(%inner, %outer) -> !llvm.ptr
    %map_middle = omp.map.info var_ptr(%middle : !llvm.ptr, i64) map_clauses(to) capture(ByRef) -> !llvm.ptr
    %map_strided = omp.map.info var_ptr(%strided : !llvm.ptr, i64) map_clauses(from) capture(ByRef) bounds(%strided_bound) -> !llvm.ptr
    %map_last = omp.map.info var_ptr(%last : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr
    omp.target_update map_entries(%map_first, %map_array, %map_middle, %map_strided, %map_last : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) nowait
    llvm.return
  }
}

// CHECK: %struct.[[$TASK_TYPE:[^ ]+]] = type { %struct.kmp_task_ompbuilder_t, %struct.[[$PRIVATES:[^ ]+]] }
// CHECK: %struct.[[$PRIVATES]] = type { [5 x ptr], [5 x ptr], [3 x %struct.descriptor_dim], [2 x %struct.descriptor_dim] }
// CHECK: @.offload_sizes = {{.*}}[5 x i64] [i64 4, i64 3, i64 8, i64 2, i64 4]
// CHECK: @.offload_maptypes = {{.*}}[5 x i64] [i64 1, i64 17592186044417, i64 1, i64 17592186044418, i64 2]
// CHECK-LABEL: define void @deferred_update(
// CHECK: %[[BASES:.*]] = alloca [5 x ptr]
// CHECK: %[[POINTERS:.*]] = alloca [5 x ptr]
// CHECK: %[[DESC0:.*]] = alloca [3 x %struct.descriptor_dim]
// CHECK: %[[DESC1:.*]] = alloca [2 x %struct.descriptor_dim]
// CHECK: store ptr %[[DESC1]],
// CHECK: %[[BASES_BEGIN:.*]] = getelementptr inbounds [5 x ptr], ptr %[[BASES]], i32 0, i32 0
// CHECK: %[[POINTERS_BEGIN:.*]] = getelementptr inbounds [5 x ptr], ptr %[[POINTERS]], i32 0, i32 0
// CHECK: %[[TASK:.*]] = call ptr @__kmpc_omp_target_task_alloc({{.*}}i64 240, i64 0, ptr @[[PROXY:[^,]+]], i64 -1)
// CHECK: %[[PRIVATE_DATA:.*]] = getelementptr inbounds nuw %struct.[[$TASK_TYPE]], ptr %[[TASK]], i32 0, i32 1
// CHECK: %[[PRIVATE_BASES:.*]] = getelementptr inbounds nuw %struct.[[$PRIVATES]], ptr %[[PRIVATE_DATA]], i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[PRIVATE_BASES]], ptr align 1 %[[BASES_BEGIN]], i64 40, i1 false)
// CHECK: %[[PRIVATE_POINTERS:.*]] = getelementptr inbounds nuw %struct.[[$PRIVATES]], ptr %[[PRIVATE_DATA]], i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[PRIVATE_POINTERS]], ptr align 1 %[[POINTERS_BEGIN]], i64 40, i1 false)
// CHECK: %[[PRIVATE_DESC0:.*]] = getelementptr inbounds nuw %struct.[[$PRIVATES]], ptr %[[PRIVATE_DATA]], i32 0, i32 2
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[PRIVATE_DESC0]], ptr align 8 %[[DESC0]], i64 72, i1 false)
// CHECK: %[[ENTRY1:.*]] = getelementptr inbounds [5 x ptr], ptr %[[PRIVATE_POINTERS]], i32 0, i32 1
// CHECK-NEXT: store ptr %[[PRIVATE_DESC0]], ptr %[[ENTRY1]]
// CHECK: %[[PRIVATE_DESC1:.*]] = getelementptr inbounds nuw %struct.[[$PRIVATES]], ptr %[[PRIVATE_DATA]], i32 0, i32 3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[PRIVATE_DESC1]], ptr align 8 %[[DESC1]], i64 48, i1 false)
// CHECK: %[[ENTRY3:.*]] = getelementptr inbounds [5 x ptr], ptr %[[PRIVATE_POINTERS]], i32 0, i32 3
// CHECK-NEXT: store ptr %[[PRIVATE_DESC1]], ptr %[[ENTRY3]]
// CHECK: call i32 @__kmpc_omp_task({{.*}}ptr %[[TASK]])
// CHECK: ret void

// CHECK: define internal void @[[WORKER:[^(]+]](i32 %{{.*}}, ptr %[[WORKER_BASES:.*]], ptr %[[WORKER_POINTERS:.*]]) {
// CHECK: call void @__tgt_target_data_update_nowait_mapper({{.*}}i64 -1, i32 5, ptr %[[WORKER_BASES]], ptr %[[WORKER_POINTERS]], ptr @.offload_sizes, ptr @.offload_maptypes,
// CHECK: ret void
// CHECK: define internal void @[[PROXY]](i32 %[[TID:.*]], ptr %[[TASK_ARG:.*]]) {
// CHECK: %[[PROXY_PRIVATE_DATA:.*]] = getelementptr inbounds nuw %struct.[[$TASK_TYPE]], ptr %[[TASK_ARG]], i32 0, i32 1
// CHECK: %[[PROXY_BASES:.*]] = getelementptr inbounds nuw %struct.[[$PRIVATES]], ptr %[[PROXY_PRIVATE_DATA]], i32 0, i32 0
// CHECK: %[[PROXY_POINTERS:.*]] = getelementptr inbounds nuw %struct.[[$PRIVATES]], ptr %[[PROXY_PRIVATE_DATA]], i32 0, i32 1
// CHECK: call void @[[WORKER]](i32 %[[TID]], ptr %[[PROXY_BASES]], ptr %[[PROXY_POINTERS]])
// CHECK: ret void

// -----

// Synchronous updates can consume the encounter-time descriptor directly.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @synchronous_update(%strided: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %bound = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c3 : i64) stride(%c16 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %map = omp.map.info var_ptr(%strided : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%bound) -> !llvm.ptr
    omp.target_update map_entries(%map : !llvm.ptr)
    llvm.return
  }
}

// CHECK: @.offload_sizes = {{.*}}[1 x i64] [i64 2]
// CHECK: @.offload_maptypes = {{.*}}[1 x i64] [i64 17592186044417]
// CHECK-LABEL: define void @synchronous_update(
// CHECK-NOT: @__kmpc
// CHECK: alloca [1 x ptr]
// CHECK: %[[SYNC_POINTERS:.*]] = alloca [1 x ptr]
// CHECK: %[[SYNC_DESC:.*]] = alloca [2 x %struct.descriptor_dim]
// CHECK-NOT: @__kmpc
// CHECK-NOT: @llvm.memcpy
// CHECK: getelementptr inbounds [2 x %struct.descriptor_dim], ptr %[[SYNC_DESC]], i64 0, i64 1
// CHECK: %[[SYNC_ENTRY:.*]] = getelementptr inbounds [1 x ptr], ptr %[[SYNC_POINTERS]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[SYNC_DESC]], ptr %[[SYNC_ENTRY]]
// CHECK-NOT: @__kmpc
// CHECK-NOT: @llvm.memcpy
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}ptr @.offload_sizes, ptr @.offload_maptypes,
// CHECK-NEXT: ret void
