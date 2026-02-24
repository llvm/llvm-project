// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_arrive_drop_expect_tx_generic(%barrier: !llvm.ptr, %txcount : i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_drop_expect_tx_generic(ptr %0, i32 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cta(ptr addrspace(3) %3, i32 %1)
  // CHECK-NEXT: %5 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %6 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cta(ptr addrspace(3) %5, i32 %1)
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cluster.space.cta(ptr addrspace(3) %7, i32 %1)
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cta.space.cta(ptr addrspace(3) %9, i32 %1)
  // CHECK-NEXT: %11 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %12 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cta.space.cta(ptr addrspace(3) %11, i32 %1)
  // CHECK-NEXT: %13 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %14 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cluster.space.cta(ptr addrspace(3) %13, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount : !llvm.ptr, i32 -> i64
  %1 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cta>} : !llvm.ptr, i32 -> i64
  %2 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32 -> i64

  %3 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {relaxed = true} : !llvm.ptr, i32 -> i64
  %4 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cta>, relaxed = true} : !llvm.ptr, i32 -> i64
  %5 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cluster>, relaxed = true} : !llvm.ptr, i32 -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_drop_expect_tx_shared(%barrier: !llvm.ptr<3>, %txcount : i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_drop_expect_tx_shared(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: %3 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %4 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %5 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %6 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %7 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %8 = call i64 @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount : !llvm.ptr<3>, i32 -> i64
  %1 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<3>, i32 -> i64
  %2 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32 -> i64

  %3 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {relaxed = true} : !llvm.ptr<3>, i32 -> i64
  %4 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cta>, relaxed = true} : !llvm.ptr<3>, i32 -> i64
  %5 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cluster>, relaxed = true} : !llvm.ptr<3>, i32 -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_drop_expect_tx_shared_cluster(%barrier: !llvm.ptr<7>, %txcount : i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_drop_expect_tx_shared_cluster(ptr addrspace(7) %0, i32 %1) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.relaxed.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount : !llvm.ptr<7>, i32
  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<7>, i32
  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<7>, i32

  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {relaxed = true} : !llvm.ptr<7>, i32
  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cta>, relaxed = true} : !llvm.ptr<7>, i32
  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %txcount {scope = #nvvm.mem_scope<cluster>, relaxed = true} : !llvm.ptr<7>, i32
  llvm.return
}

