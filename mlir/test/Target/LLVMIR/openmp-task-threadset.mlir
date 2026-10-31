// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that threadset(omp_pool) sets the free-agent task flag (0x80 = 128) and
// threadset(omp_team) does not.

llvm.func @task_threadset_pool() {
  omp.task threadset(omp_pool) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @task_threadset_pool()
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 129, i64 {{.+}}, i64 {{.+}}, ptr @{{.+}})

// -----

llvm.func @task_threadset_team() {
  omp.task threadset(omp_team) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @task_threadset_team()
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 1, i64 {{.+}}, i64 {{.+}}, ptr @{{.+}})
