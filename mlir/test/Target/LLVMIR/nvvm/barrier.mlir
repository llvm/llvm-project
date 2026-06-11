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
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 %[[numThreads]])
  // CHECK: nvvm.barrier number_of_threads = %{{.*}}
  nvvm.barrier number_of_threads = %numberOfThreads
  // LLVM: %[[redOperandCmp1:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.and.aligned.all(i32 0, i1 %[[redOperandCmp1]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<and> %{{.*}} -> i32
  %0 = nvvm.barrier.reduction #nvvm.reduction<and> %redOperand -> i32
  // LLVM: %[[redOperandCmp2:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.or.aligned.all(i32 0, i1 %[[redOperandCmp2]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<or> %{{.*}} -> i32
  %1 = nvvm.barrier.reduction #nvvm.reduction<or> %redOperand -> i32
  // LLVM: %[[redOperandCmp3:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i32 @llvm.nvvm.barrier.cta.red.popc.aligned.all(i32 0, i1 %[[redOperandCmp3]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<popc> %{{.*}} -> i32
  %2 = nvvm.barrier.reduction #nvvm.reduction<popc> %redOperand -> i32
  // LLVM: %[[redOperandCmp4:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.and.aligned.all(i32 %[[barId]], i1 %[[redOperandCmp4]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<and> %{{.*}} id = %{{.*}} -> i32
  %3 = nvvm.barrier.reduction #nvvm.reduction<and> %redOperand id = %barID -> i32

  // Non-aligned sync variants (opt-out).
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.all(i32 0)
  // CHECK: nvvm.barrier {aligned = false}
  nvvm.barrier {aligned = false}
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.count(i32 %[[barId]], i32 %[[numThreads]])
  // CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}} {aligned = false}
  nvvm.barrier id = %barID number_of_threads = %numberOfThreads {aligned = false}

  // Non-aligned reduction variants (opt-out).
  // LLVM: %[[redOperandCmp5:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.and.all(i32 0, i1 %[[redOperandCmp5]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<and> %{{.*}} -> i32 {aligned = false}
  %4 = nvvm.barrier.reduction #nvvm.reduction<and> %redOperand -> i32 {aligned = false}
  // LLVM: %[[redOperandCmp6:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i1 @llvm.nvvm.barrier.cta.red.or.all(i32 0, i1 %[[redOperandCmp6]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<or> %{{.*}} -> i32 {aligned = false}
  %5 = nvvm.barrier.reduction #nvvm.reduction<or> %redOperand -> i32 {aligned = false}
  // LLVM: %[[redOperandCmp7:.*]] = icmp ne i32 %[[redOperand]], 0
  // LLVM: %{{.*}} = call i32 @llvm.nvvm.barrier.cta.red.popc.all(i32 0, i1 %[[redOperandCmp7]])
  // CHECK: %{{.*}} = nvvm.barrier.reduction #nvvm.reduction<popc> %{{.*}} -> i32 {aligned = false}
  %6 = nvvm.barrier.reduction #nvvm.reduction<popc> %redOperand -> i32 {aligned = false}

  llvm.return
}
