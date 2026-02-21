// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["nvptx64-nvidia-cuda"]} {
  llvm.func @_QPopenmp_target(%d16 : i16, %d32 : i32, %d64 : i64) {
    %x  = llvm.mlir.constant(0 : i32) : i32

    // Constant i16 -> i64 in the runtime call.
    %c1_i16 = llvm.mlir.constant(1 : i16) : i16
    omp.target device(%c1_i16 : i16)
      host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.terminator
    }

    // Constant i32 -> i64 in the runtime call.
    %c2_i32 = llvm.mlir.constant(2 : i32) : i32
    omp.target device(%c2_i32 : i32)
      host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.terminator
    }

    // Constant i64 stays i64 in the runtime call.
    %c3_i64 = llvm.mlir.constant(3 : i64) : i64
    omp.target device(%c3_i64 : i64)
      host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.terminator
    }

    // Variable i16 -> cast to i64.
    omp.target device(%d16 : i16)
      host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.terminator
    }

    // Variable i32 -> cast to i64.
    omp.target device(%d32 : i32)
      host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.terminator
    }

    // Variable i64 stays i64.
    omp.target device(%d64 : i64)
      host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.terminator
    }

    llvm.return
  }
}

// CHECK-LABEL: define void @_QPopenmp_target(i16 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}) {
// CHECK: br label %entry
// CHECK: entry:

// ---- Constant cases (device id is 2nd argument) ----
// CHECK-DAG: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 1, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 2, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 3, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i16 -> i64
// CHECK: %[[D16_I64:.*]] = sext i16 %{{.*}} to i64
// CHECK: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 %[[D16_I64]], i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i32 -> i64
// CHECK: %[[D32_I64:.*]] = sext i32 %{{.*}} to i64
// CHECK: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 %[[D32_I64]], i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i64
// CHECK: call i32 @__tgt_target_kernel(ptr {{.*}}, i64 %{{.*}}, i32 {{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}})

// -----

module attributes {omp.is_target_device = false, omp.target_triples = ["nvptx64-nvidia-cuda"]} {
  llvm.func @_QPopenmp_target_data(%d16 : i16, %d32 : i32, %d64 : i64) {
    %one = llvm.mlir.constant(1 : i64) : i64
    %buf = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %map = omp.map.info var_ptr(%buf : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}

    // Constant i16 -> i64 in the runtime call.
    %c1_i16 = llvm.mlir.constant(1 : i16) : i16
    omp.target_data device(%c1_i16 : i16) map_entries(%map : !llvm.ptr) {
      omp.terminator
    }

    // Constant i32 -> i64 in the runtime call.
    %c2_i32 = llvm.mlir.constant(2 : i32) : i32
    omp.target_data device(%c2_i32 : i32) map_entries(%map : !llvm.ptr) {
      omp.terminator
    }

    // Constant i64 stays i64 in the runtime call.
    %c3_i64 = llvm.mlir.constant(3 : i64) : i64
    omp.target_data device(%c3_i64 : i64) map_entries(%map : !llvm.ptr) {
      omp.terminator
    }

    // Variable i16 -> cast to i64.
    omp.target_data device(%d16 : i16) map_entries(%map : !llvm.ptr) {
      omp.terminator
    }

    // Variable i32 -> cast to i64.
    omp.target_data device(%d32 : i32) map_entries(%map : !llvm.ptr) {
      omp.terminator
    }

    // Variable i64 stays i64.
    omp.target_data device(%d64 : i64) map_entries(%map : !llvm.ptr) {
      omp.terminator
    }

    llvm.return
  }
}

// CHECK-LABEL: define void @_QPopenmp_target_data(i16 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}) {
// CHECK: br label %entry
// CHECK: entry:

// ---- Constant cases (device id is 2nd argument) ----
// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)
// CHECK-DAG: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)

// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 2, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)
// CHECK-DAG: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 2, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)

// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 3, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)
// CHECK-DAG: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 3, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)

// Variable i16 -> i64
// CHECK: %[[D16_I64:.*]] = sext i16 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 %[[D16_I64]], i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)
// CHECK: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 %[[D16_I64]], i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)

// Variable i32 -> i64
// CHECK: %[[D32_I64:.*]] = sext i32 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 %[[D32_I64]], i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)
// CHECK: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 %[[D32_I64]], i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)

// Variable i64
// CHECK: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 %{{.*}}, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)
// CHECK: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 %{{.*}}, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null)

// -----

module attributes {omp.is_target_device = false, omp.target_triples = ["nvptx64-nvidia-cuda"]} {
  llvm.func @_QPomp_target_enter_exit(%d16 : i16, %d32 : i32, %d64 : i64) {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %var = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr

    %m_to = omp.map.info var_ptr(%var : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "var"}
    %m_from = omp.map.info var_ptr(%var : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = "var"}

    // Constant i16 -> i64 in the runtime call.
    %c1_i16 = llvm.mlir.constant(1 : i16) : i16
    omp.target_enter_data device(%c1_i16 : i16) map_entries(%m_to : !llvm.ptr)

    // Constant i32 -> i64 in the runtime call.
    %c2_i32 = llvm.mlir.constant(2 : i32) : i32
    omp.target_enter_data device(%c2_i32 : i32) map_entries(%m_to : !llvm.ptr)

    // Constant i64 stays i64 in the runtime call.
    %c3_i64 = llvm.mlir.constant(3 : i64) : i64
    omp.target_enter_data device(%c3_i64 : i64) map_entries(%m_to : !llvm.ptr)

    // ---- Variable cases (enter) ----
    omp.target_enter_data device(%d16 : i16) map_entries(%m_to : !llvm.ptr)
    omp.target_enter_data device(%d32 : i32) map_entries(%m_to : !llvm.ptr)
    omp.target_enter_data device(%d64 : i64) map_entries(%m_to : !llvm.ptr)

    // ---- Constant cases (exit) ----
    omp.target_exit_data device(%c1_i16 : i16) map_entries(%m_from : !llvm.ptr)
    omp.target_exit_data device(%c2_i32 : i32) map_entries(%m_from : !llvm.ptr)
    omp.target_exit_data device(%c3_i64 : i64) map_entries(%m_from : !llvm.ptr)

    // ---- Variable cases (exit) ----
    omp.target_exit_data device(%d16 : i16) map_entries(%m_from : !llvm.ptr)
    omp.target_exit_data device(%d32 : i32) map_entries(%m_from : !llvm.ptr)
    omp.target_exit_data device(%d64 : i64) map_entries(%m_from : !llvm.ptr)

    llvm.return
  }
}

// CHECK-LABEL: define void @_QPomp_target_enter_exit(i16 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}) {
// CHECK: br label %entry
// CHECK: entry:

// ---- Constant enter cases (device id is 2nd argument) ----
// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 1, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 2, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 3, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// ---- Variable enter cases ----
// Variable i16 -> i64
// CHECK: %[[D16_I64_BEGIN:.*]] = sext i16 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 %[[D16_I64_BEGIN]], i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i32 -> i64
// CHECK: %[[D32_I64_BEGIN:.*]] = sext i32 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 %[[D32_I64_BEGIN]], i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i64 stays i64
// CHECK: call void @__tgt_target_data_begin_mapper(ptr {{.*}}, i64 %{{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// ---- Constant exit cases (device id is 2nd argument) ----
// CHECK-DAG: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 1, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 2, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 3, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// ---- Variable exit cases ----
// Variable i16 -> i64
// CHECK: %[[D16_I64_END:.*]] = sext i16 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 %[[D16_I64_END]], i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i32 -> i64
// CHECK: %[[D32_I64_END:.*]] = sext i32 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 %[[D32_I64_END]], i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i64 stays i64
// CHECK: call void @__tgt_target_data_end_mapper(ptr {{.*}}, i64 %{{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// CHECK: ret void
// CHECK: }

// -----

module attributes {omp.is_target_device = false, omp.target_triples = ["nvptx64-nvidia-cuda"]} {
  llvm.func @target_update_dev_clause(%d16 : i16, %d32 : i32, %d64 : i64) {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %var = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
    %m = omp.map.info var_ptr(%var : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "var"}

    // ---- Constant cases ----
    %c1_i16 = llvm.mlir.constant(1 : i16) : i16
    omp.target_update device(%c1_i16 : i16) map_entries(%m : !llvm.ptr)

    %c2_i32 = llvm.mlir.constant(2 : i32) : i32
    omp.target_update device(%c2_i32 : i32) map_entries(%m : !llvm.ptr)

    %c3_i64 = llvm.mlir.constant(3 : i64) : i64
    omp.target_update device(%c3_i64 : i64) map_entries(%m : !llvm.ptr)

    // ---- Variable cases ----
    omp.target_update device(%d16 : i16) map_entries(%m : !llvm.ptr)
    omp.target_update device(%d32 : i32) map_entries(%m : !llvm.ptr)
    omp.target_update device(%d64 : i64) map_entries(%m : !llvm.ptr)

    llvm.return
  }
}

// CHECK-LABEL: define void @target_update_dev_clause(i16 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}) {
// CHECK: br label %entry
// CHECK: entry:

// ---- Constant cases (device id is 2nd argument) ----
// CHECK-DAG: call void @__tgt_target_data_update_mapper(ptr {{.*}}, i64 1, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call void @__tgt_target_data_update_mapper(ptr {{.*}}, i64 2, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DAG: call void @__tgt_target_data_update_mapper(ptr {{.*}}, i64 3, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// ---- Variable cases ----
// Variable i16 -> i64
// CHECK: %[[D16_I64:.*]] = sext i16 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_update_mapper(ptr {{.*}}, i64 %[[D16_I64]], i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i32 -> i64
// CHECK: %[[D32_I64:.*]] = sext i32 %{{.*}} to i64
// CHECK: call void @__tgt_target_data_update_mapper(ptr {{.*}}, i64 %[[D32_I64]], i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// Variable i64 stays i64
// CHECK: call void @__tgt_target_data_update_mapper(ptr {{.*}}, i64 %{{.*}}, i32 {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})

// CHECK: ret void
// CHECK: }
