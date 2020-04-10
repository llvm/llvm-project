// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @empty()
llvm.func @empty() {
  // CHECK: [[OMP_THREAD:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK-NEXT:  call void @__kmpc_barrier(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD]])
  omp.barrier

  // CHECK: [[OMP_THREAD1:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK-NEXT:  [[RET_VAL:%.*]] = call i32 @__kmpc_omp_taskwait(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD1]])
  omp.taskwait

  // CHECK: [[OMP_THREAD2:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{[0-9]+}})
  // CHECK-NEXT:  [[RET_VAL:%.*]] = call i32 @__kmpc_omp_taskyield(%struct.ident_t* @{{[0-9]+}}, i32 [[OMP_THREAD2]], i32 0)
  omp.taskyield

// CHECK-NEXT:    ret void
  llvm.return
}
