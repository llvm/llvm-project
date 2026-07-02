// RUN: mlir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @taskwait_depend(%x: !llvm.ptr) {
  omp.taskwait depend(taskdependout -> %x : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @taskwait_depend
// CHECK-SAME: (ptr[[xaddr:.+]])
// CHECK: %[[dep_arr_addr:.+]] = alloca [1 x %struct.kmp_dep_info], align 8
// CHECK: %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num({{.+}})
// CHECK: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 1, ptr %[[dep_arr_addr]], i32 0, ptr null, i32 0)
