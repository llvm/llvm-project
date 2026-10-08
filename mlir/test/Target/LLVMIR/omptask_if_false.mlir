// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @foo_(%arg0: !llvm.ptr {fir.bindc_name = "n"}, %arg1: !llvm.ptr {fir.bindc_name = "r"}) attributes {fir.internal_name = "_QPfoo"} {
  %0 = llvm.mlir.constant(false) : i1
  omp.task if(%0) depend(taskdependin -> %arg0 : !llvm.ptr) {
    %1 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %1, %arg1 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK: call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 5, i64 40, i64 16, ptr @foo_..omp_par)
// CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.+}}, i32 %{{.+}}, ptr %{{.+}}, i32 1, ptr %{{.+}}, i32 0, ptr null)

// CHECK-LABEL: define void @foo_detach_if_false
llvm.func @foo_detach_if_false(%arg0: !llvm.ptr {fir.bindc_name = "n"}, %arg1: !llvm.ptr {fir.bindc_name = "ev"}) {
  %0 = llvm.mlir.constant(false) : i1
  omp.task if(%0) depend(taskdependin -> %arg0 : !llvm.ptr) detach(%arg1 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK: %[[TID:.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK: %[[TASK:.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %[[TID]], i32 69, i64 40, i64 0, ptr @foo_detach_if_false..omp_par)
// CHECK: %[[EVENT:.+]] = call ptr @__kmpc_task_allow_completion_event(ptr @{{.+}}, i32 %[[TID]], ptr %[[TASK]])
// CHECK: %[[INT_EVENT:.+]] = ptrtoint ptr %[[EVENT]] to i64
// CHECK: store i64 %[[INT_EVENT]], ptr %{{.+}}, align 4
// CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.+}}, i32 %[[TID]], ptr %[[TASK]], i32 1, ptr %{{.+}}, i32 0, ptr null)
// CHECK-NOT: @__kmpc_omp_task_begin_if0
// CHECK-NOT: @__kmpc_omp_task_complete_if0

