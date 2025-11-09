// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_barrier(
// CHECK-SAME: i32 %[[barId:.*]], i32 %[[numThreads:.*]], i32 %[[predicate:.*]])
llvm.func @llvm_nvvm_barrier(%barID : i32, %numberOfThreads : i32, %predicate : i32) {
  // CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  nvvm.barrier
  // CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %[[barId]])
  nvvm.barrier id = %barID
  // CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %[[barId]], i32 %[[numThreads]])
  nvvm.barrier id = %barID number_of_threads = %numberOfThreads
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.barrier0.and(i32 %[[predicate]])
  %0 = nvvm.barrier #nvvm.reduction<and> %predicate -> i32
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.barrier0.or(i32 %[[predicate]])
  %1 = nvvm.barrier #nvvm.reduction<or> %predicate -> i32
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.barrier0.popc(i32 %[[predicate]])
  %2 = nvvm.barrier #nvvm.reduction<popc> %predicate -> i32

  llvm.return
}
