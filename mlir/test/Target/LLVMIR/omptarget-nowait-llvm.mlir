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


// CHECK: define void @_QPfoo() {

// CHECK:   %[[TASK:.*]] = call ptr @__kmpc_omp_target_task_alloc
// CHECK-SAME:     (ptr @{{.*}}, i32 %{{.*}}, i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, ptr
// CHECK-SAME:     @[[TASK_PROXY_FUNC:.*]], i64 {{.*}})

// CHECK:   call i32 @__kmpc_omp_task(ptr {{.*}}, i32 %{{.*}}, ptr %[[TASK]])
// CHECK: }


// CHECK: define internal void @[[TASK_PROXY_FUNC]](i32 %{{.*}}, ptr %{{.*}}) {
// CHECK:   call void @_QPfoo..omp_par(i32 %{{.*}}, ptr %{{.*}})
// CHECK: }
}
