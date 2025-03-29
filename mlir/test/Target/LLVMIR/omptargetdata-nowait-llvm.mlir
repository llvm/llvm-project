// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s 2>&1 | FileCheck %s

llvm.func @_QPopenmp_target_data_enter() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}

  omp.target_enter_data map_entries(%2 : !llvm.ptr) nowait

  llvm.return
}

// CHECK: define void @_QPopenmp_target_data_enter() {

// CHECK:   %[[TASK:.*]] = call ptr @__kmpc_omp_target_task_alloc
// CHECK-SAME:     (ptr @{{.*}}, i32 %{{.*}}, i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, ptr
// CHECK-SAME:     @[[TASK_PROXY_FUNC:.*]], i64 {{.*}})

// CHECK:   call i32 @__kmpc_omp_task(ptr {{.*}}, i32 %{{.*}}, ptr %[[TASK]])
// CHECK: }

// CHECK: define internal void @[[TASK_BODY_FUNC:.*]](i32 %[[TID:.*]], ptr %[[TASK_ARG:.*]]) {
// CHECK:   %[[OFFLOAD_BASE_PTRS:.*]] = getelementptr { ptr, ptr }, ptr %[[TASK_ARG]], i32 0, i32 0
// CHECK:   %[[OFFLOAD_BASE_PTRS_VAL:.*]] = load ptr, ptr %[[OFFLOAD_BASE_PTRS]], align 8
// CHECK:   %[[OFFLOAD_PTRS:.*]] = getelementptr { ptr, ptr }, ptr %[[TASK_ARG]], i32 0, i32 1
// CHECK:   %[[OFFLOAD_PTRS_VAL:.*]] = load ptr, ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  call void @__tgt_target_data_begin_nowait_mapper(
// CHECK-SAME: ptr @{{.*}}, i64 -1, i32 1,
// CHECK-SAME: ptr %[[OFFLOAD_BASE_PTRS_VAL]], ptr %[[OFFLOAD_PTRS_VAL]],
// CHECK-SAME: ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr null, i32 0, ptr null, i32 0, ptr null)
// CHECK: }

// CHECK: define internal void @[[TASK_PROXY_FUNC]](i32 %{{.*}}, ptr %{{.*}}) {
// CHECK:   call void @[[TASK_BODY_FUNC]](i32 %{{.*}}, ptr %{{.*}})
// CHECK: }

// -----

llvm.func @_QPopenmp_target_data_update() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}

  omp.target_update map_entries(%2 : !llvm.ptr) nowait

  llvm.return
}

// CHECK: define void @_QPopenmp_target_data_update() {

// CHECK:   %[[TASK:.*]] = call ptr @__kmpc_omp_target_task_alloc
// CHECK-SAME:     (ptr @{{.*}}, i32 %{{.*}}, i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, ptr
// CHECK-SAME:     @[[TASK_PROXY_FUNC:.*]], i64 {{.*}})

// CHECK:   call i32 @__kmpc_omp_task(ptr {{.*}}, i32 %{{.*}}, ptr %[[TASK]])
// CHECK: }

// CHECK: define internal void @[[TASK_BODY_FUNC:.*]](i32 %[[TID:.*]], ptr %[[TASK_ARG:.*]]) {
// CHECK:   %[[OFFLOAD_BASE_PTRS:.*]] = getelementptr { ptr, ptr }, ptr %[[TASK_ARG]], i32 0, i32 0
// CHECK:   %[[OFFLOAD_BASE_PTRS_VAL:.*]] = load ptr, ptr %[[OFFLOAD_BASE_PTRS]], align 8
// CHECK:   %[[OFFLOAD_PTRS:.*]] = getelementptr { ptr, ptr }, ptr %[[TASK_ARG]], i32 0, i32 1
// CHECK:   %[[OFFLOAD_PTRS_VAL:.*]] = load ptr, ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  call void @__tgt_target_data_update_nowait_mapper(
// CHECK-SAME: ptr @{{.*}}, i64 -1, i32 1,
// CHECK-SAME: ptr %[[OFFLOAD_BASE_PTRS_VAL]], ptr %[[OFFLOAD_PTRS_VAL]],
// CHECK-SAME: ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr null, i32 0, ptr null, i32 0, ptr null)
// CHECK: }

// CHECK: define internal void @[[TASK_PROXY_FUNC]](i32 %{{.*}}, ptr %{{.*}}) {
// CHECK:   call void @[[TASK_BODY_FUNC]](i32 %{{.*}}, ptr %{{.*}})
// CHECK: }

// -----

llvm.func @_QPopenmp_target_data_exit() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}

  omp.target_exit_data map_entries(%2 : !llvm.ptr) nowait

  llvm.return
}

// CHECK: define void @_QPopenmp_target_data_exit() {

// CHECK:   %[[TASK:.*]] = call ptr @__kmpc_omp_target_task_alloc
// CHECK-SAME:     (ptr @{{.*}}, i32 %{{.*}}, i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, ptr
// CHECK-SAME:     @[[TASK_PROXY_FUNC:.*]], i64 {{.*}})

// CHECK:   call i32 @__kmpc_omp_task(ptr {{.*}}, i32 %{{.*}}, ptr %[[TASK]])
// CHECK: }

// CHECK: define internal void @[[TASK_BODY_FUNC:.*]](i32 %[[TID:.*]], ptr %[[TASK_ARG:.*]]) {
// CHECK:   %[[OFFLOAD_BASE_PTRS:.*]] = getelementptr { ptr, ptr }, ptr %[[TASK_ARG]], i32 0, i32 0
// CHECK:   %[[OFFLOAD_BASE_PTRS_VAL:.*]] = load ptr, ptr %[[OFFLOAD_BASE_PTRS]], align 8
// CHECK:   %[[OFFLOAD_PTRS:.*]] = getelementptr { ptr, ptr }, ptr %[[TASK_ARG]], i32 0, i32 1
// CHECK:   %[[OFFLOAD_PTRS_VAL:.*]] = load ptr, ptr %[[OFFLOAD_PTRS]], align 8

// CHECK:  call void @__tgt_target_data_end_nowait_mapper(
// CHECK-SAME: ptr @{{.*}}, i64 -1, i32 1,
// CHECK-SAME: ptr %[[OFFLOAD_BASE_PTRS_VAL]], ptr %[[OFFLOAD_PTRS_VAL]],
// CHECK-SAME: ptr @{{.*}}, ptr @{{.*}}, ptr @{{.*}}, ptr null, i32 0, ptr null, i32 0, ptr null)
// CHECK: }

// CHECK: define internal void @[[TASK_PROXY_FUNC]](i32 %{{.*}}, ptr %{{.*}}) {
// CHECK:   call void @[[TASK_BODY_FUNC]](i32 %{{.*}}, ptr %{{.*}})
// CHECK: }
