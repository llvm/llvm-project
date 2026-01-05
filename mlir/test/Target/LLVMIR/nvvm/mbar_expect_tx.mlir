// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_expect_tx_shared(%barrier: !llvm.ptr<3>, %tx_count : i32) {
  // CHECK-LABEL: define void @mbarrier_expect_tx_shared(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.expect_tx %barrier, %tx_count : !llvm.ptr<3>, i32
  nvvm.mbarrier.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<3>, i32
  nvvm.mbarrier.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32

  llvm.return
}

llvm.func @mbarrier_expect_tx_shared_cluster(%barrier: !llvm.ptr<7>, %tx_count : i32) {
  // CHECK-LABEL: define void @mbarrier_expect_tx_shared_cluster(ptr addrspace(7) %0, i32 %1) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.expect_tx %barrier, %tx_count : !llvm.ptr<7>, i32
  nvvm.mbarrier.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<7>, i32
  nvvm.mbarrier.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<7>, i32

  llvm.return
}