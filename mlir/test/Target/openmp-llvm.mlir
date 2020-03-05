// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @empty()
// CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
// CHECK-NEXT:  call void @__kmpc_barrier(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD]])
// CHECK-NEXT:    ret void
llvm.func @empty() {
  omp.barrier
  llvm.return
}
