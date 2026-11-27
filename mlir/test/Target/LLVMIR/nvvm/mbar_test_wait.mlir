// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_test_wait_state(%barrier: !llvm.ptr, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_test_wait_state(ptr %0, i64 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.scope.cta.space.cta(ptr addrspace(3) %3, i64 %1)
  // CHECK-NEXT: %5 = extractvalue { i1, i1, i32 } %4, 0
  // CHECK-NEXT: %6 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.scope.cluster.space.cta(ptr addrspace(3) %6, i64 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %9, i64 %1)
  // CHECK-NEXT: %11 = extractvalue { i1, i1, i32 } %10, 0
  // CHECK-NEXT: %12 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %13 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %12, i64 %1)
  // CHECK-NEXT: %14 = extractvalue { i1, i1, i32 } %13, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %state : !llvm.ptr, i64 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %state  scope = cluster : !llvm.ptr, i64 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %state  relaxed = true : !llvm.ptr, i64 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %state  scope = cluster relaxed = true : !llvm.ptr, i64 -> i1
  llvm.return
}

llvm.func @mbarrier_test_wait_shared_state(%barrier: !llvm.ptr<3>, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_test_wait_shared_state(ptr addrspace(3) %0, i64 %1) {
  // CHECK-NEXT: %3 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %4 = extractvalue { i1, i1, i32 } %3, 0
  // CHECK-NEXT: %5 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %6 = extractvalue { i1, i1, i32 } %5, 0
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %10 = extractvalue { i1, i1, i32 } %9, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %state : !llvm.ptr<3>, i64 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %state  scope = cluster : !llvm.ptr<3>, i64 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %state  relaxed = true : !llvm.ptr<3>, i64 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %state  scope = cluster relaxed = true : !llvm.ptr<3>, i64 -> i1
  llvm.return
}

llvm.func @mbarrier_test_wait_phase(%barrier: !llvm.ptr, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_test_wait_phase(ptr %0, i32 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.scope.cta.space.cta(ptr addrspace(3) %3, i32 %1)
  // CHECK-NEXT: %5 = extractvalue { i1, i1, i32 } %4, 0
  // CHECK-NEXT: %6 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.scope.cluster.space.cta(ptr addrspace(3) %6, i32 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %9, i32 %1)
  // CHECK-NEXT: %11 = extractvalue { i1, i1, i32 } %10, 0
  // CHECK-NEXT: %12 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %13 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %12, i32 %1)
  // CHECK-NEXT: %14 = extractvalue { i1, i1, i32 } %13, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %phase : !llvm.ptr, i32 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %phase  scope = cluster : !llvm.ptr, i32 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %phase  relaxed = true : !llvm.ptr, i32 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %phase  scope = cluster relaxed = true : !llvm.ptr, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_test_wait_shared_phase(%barrier: !llvm.ptr<3>, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_test_wait_shared_phase(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: %3 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %4 = extractvalue { i1, i1, i32 } %3, 0
  // CHECK-NEXT: %5 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %6 = extractvalue { i1, i1, i32 } %5, 0
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.test.wait.parity.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %10 = extractvalue { i1, i1, i32 } %9, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %phase : !llvm.ptr<3>, i32 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %phase  scope = cluster : !llvm.ptr<3>, i32 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %phase  relaxed = true : !llvm.ptr<3>, i32 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %phase  scope = cluster relaxed = true : !llvm.ptr<3>, i32 -> i1
  llvm.return
}
