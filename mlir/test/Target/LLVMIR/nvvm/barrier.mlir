// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// LLVM-LABEL: @llvm_nvvm_barrier(
// LLVM-SAME: i32 %[[barId:.*]], i32 %[[numThreads:.*]], i32 %[[redOperand:.*]])
llvm.func @llvm_nvvm_barrier(%barID : i32, %numberOfThreads : i32, %redOperand : i32) {
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  // CHECK: nvvm.barrier
  nvvm.barrier
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %[[barId]])
  // CHECK: nvvm.barrier id = %{{.*}}
  nvvm.barrier id = %barID
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %[[barId]], i32 %[[numThreads]])
  // CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}}
  nvvm.barrier id = %barID number_of_threads = %numberOfThreads
  // LLVM: %[[redOperandCmp1:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.and.aligned.all(i32 0, i1 %[[redOperandCmp1]])
  // CHECK: %{{.*}} = nvvm.barrier #nvvm.reduction<and> %{{.*}} -> i32
  %0 = nvvm.barrier #nvvm.reduction<and> %redOperand -> i32
  // LLVM: %[[redOperandCmp2:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.or.aligned.all(i32 0, i1 %[[redOperandCmp2]])
  // CHECK: %{{.*}} = nvvm.barrier #nvvm.reduction<or> %{{.*}} -> i32
  %1 = nvvm.barrier #nvvm.reduction<or> %redOperand -> i32
  // LLVM: %[[redOperandCmp3:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i32 @llvm.nvvm.barrier.cta.red.popc.aligned.all(i32 0, i1 %[[redOperandCmp3]])
  // CHECK: %{{.*}} = nvvm.barrier #nvvm.reduction<popc> %{{.*}} -> i32
  %2 = nvvm.barrier #nvvm.reduction<popc> %redOperand -> i32

  llvm.return
}
