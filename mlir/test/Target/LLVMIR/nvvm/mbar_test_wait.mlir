// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_test_wait_state(%barrier: !llvm.ptr, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_test_wait_state(ptr %0, i64 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.test.wait.scope.cta.space.cta(ptr addrspace(3) %3, i64 %1)
  // CHECK-NEXT: %5 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.test.wait.scope.cluster.space.cta(ptr addrspace(3) %5, i64 %1)
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call i1 @llvm.nvvm.mbarrier.test.wait.relaxed.scope.cta.space.cta(ptr addrspace(3) %7, i64 %1)
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call i1 @llvm.nvvm.mbarrier.test.wait.relaxed.scope.cluster.space.cta(ptr addrspace(3) %9, i64 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %state : !llvm.ptr, i64 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %state {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i64 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %state {relaxed = true} : !llvm.ptr, i64 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %state {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i64 -> i1
  llvm.return
}

llvm.func @mbarrier_test_wait_shared_state(%barrier: !llvm.ptr<3>, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_test_wait_shared_state(ptr addrspace(3) %0, i64 %1) {
  // CHECK-NEXT: %3 = call i1 @llvm.nvvm.mbarrier.test.wait.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.test.wait.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.test.wait.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.test.wait.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %state : !llvm.ptr<3>, i64 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %state {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i64 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %state {relaxed = true} : !llvm.ptr<3>, i64 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %state {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i64 -> i1
  llvm.return
}

llvm.func @mbarrier_test_wait_phase(%barrier: !llvm.ptr, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_test_wait_phase(ptr %0, i32 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta(ptr addrspace(3) %3, i32 %1)
  // CHECK-NEXT: %5 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cluster.space.cta(ptr addrspace(3) %5, i32 %1)
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.relaxed.scope.cta.space.cta(ptr addrspace(3) %7, i32 %1)
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.relaxed.scope.cluster.space.cta(ptr addrspace(3) %9, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %phase : !llvm.ptr, i32 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %phase {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %phase {relaxed = true} : !llvm.ptr, i32 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %phase {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_test_wait_shared_phase(%barrier: !llvm.ptr<3>, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_test_wait_shared_phase(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: %3 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.test.wait.parity.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.test.wait %barrier, %phase : !llvm.ptr<3>, i32 -> i1
  %1 = nvvm.mbarrier.test.wait %barrier, %phase {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32 -> i1

  %2 = nvvm.mbarrier.test.wait %barrier, %phase {relaxed = true} : !llvm.ptr<3>, i32 -> i1
  %3 = nvvm.mbarrier.test.wait %barrier, %phase {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32 -> i1
  llvm.return
}
