// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["nvptx64-nvidia-cuda"]} {
  llvm.func @foo(%d16 : i16, %d32 : i32, %d64 : i64) {
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

// CHECK-LABEL: define void @foo(i16 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}) {
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