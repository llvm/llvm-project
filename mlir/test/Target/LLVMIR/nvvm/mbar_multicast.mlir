// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The multicast mask is only encoded by the *.space.cluster intrinsics. When it
// is absent, they still take a mask and a flag: a zero mask and a false flag,
// selecting the non-multicast form.

llvm.func @mbarrier_arrive_multicast(%barrier: !llvm.ptr<7>, %count: i32, %mc: i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_multicast(ptr addrspace(7) %0, i32 %1, i32 %2) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.arrive %barrier, %count multicast = %mc : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count multicast = %mc {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count multicast = %mc {relaxed = true} : !llvm.ptr<7>
  nvvm.mbarrier.arrive_drop %barrier, %count multicast = %mc : !llvm.ptr<7>
  llvm.return
}

llvm.func @mbarrier_arrive_expect_tx_multicast(%barrier: !llvm.ptr<7>, %tx: i32, %mc: i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_expect_tx_multicast(ptr addrspace(7) %0, i32 %1, i32 %2) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.arrive.expect_tx %barrier, %tx multicast = %mc : !llvm.ptr<7>, i32, i32
  nvvm.mbarrier.arrive_drop.expect_tx %barrier, %tx multicast = %mc : !llvm.ptr<7>, i32, i32
  llvm.return
}

llvm.func @mbarrier_tx_multicast(%barrier: !llvm.ptr<7>, %tx: i32, %mc: i32) {
  // CHECK-LABEL: define void @mbarrier_tx_multicast(ptr addrspace(7) %0, i32 %1, i32 %2) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.complete.tx.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.expect_tx %barrier, %tx multicast = %mc : !llvm.ptr<7>, i32, i32
  nvvm.mbarrier.complete_tx %barrier, %tx multicast = %mc {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<7>, i32, i32
  llvm.return
}
