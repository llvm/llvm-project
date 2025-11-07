// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_barrier0(
// CHECK-SAME: i32 %[[VALUE:.*]])
llvm.func @llvm_nvvm_barrier0(%c : i32) {
  // CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  nvvm.barrier0 
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.barrier0.and(i32 %[[VALUE]])
  %0 = nvvm.barrier0.pred %c : i32 #nvvm.barrier0_pred<and> -> i32
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.barrier0.or(i32 %[[VALUE]])
  %1 = nvvm.barrier0.pred %c : i32 #nvvm.barrier0_pred<or> -> i32
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.barrier0.popc(i32 %[[VALUE]])
  %2 = nvvm.barrier0.pred %c : i32 #nvvm.barrier0_pred<popc> -> i32
  llvm.return
}
