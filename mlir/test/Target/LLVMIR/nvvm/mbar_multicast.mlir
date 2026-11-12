// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: %if has_llc_binary && host-supports-nvptx %{ mlir-translate --mlir-to-llvmir %s | %llc-verify -o - -march=nvptx64 -mcpu=sm_107f -mattr=+ptx94 | FileCheck %s --check-prefix=PTX %}

// PTX-LABEL: multicast(
llvm.func @multicast(%bar: !llvm.ptr<7>, %count: i32, %mask: i32) {
  // LLVM: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // PTX: mbarrier.expect_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64 [{{%rd[0-9]+}}], {{%r[0-9]+}}, {{%r[0-9]+}};
  nvvm.mbarrier.expect_tx %bar, %count multicast_mask = %mask : !llvm.ptr<7>, i32, i32
  // LLVM: call void @llvm.nvvm.mbarrier.complete.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // PTX: mbarrier.complete_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64 [{{%rd[0-9]+}}], {{%r[0-9]+}}, {{%r[0-9]+}};
  nvvm.mbarrier.complete_tx %bar, %count multicast_mask = %mask : !llvm.ptr<7>, i32, i32
  // LLVM: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 1, i32 %2, i1 true)
  // PTX: mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64 _, [{{%rd[0-9]+}}], {{%r[0-9]+}}, {{%r[0-9]+}};
  nvvm.mbarrier.arrive %bar multicast_mask = %mask : !llvm.ptr<7>
  // LLVM: call void @llvm.nvvm.mbarrier.arrive.drop.scope.cta.space.cluster(ptr addrspace(7) %0, i32 1, i32 %2, i1 true)
  // PTX: mbarrier.arrive_drop.shared::cluster.multicast::cluster::32b.b64 _, [{{%rd[0-9]+}}], {{%r[0-9]+}}, {{%r[0-9]+}};
  nvvm.mbarrier.arrive_drop %bar multicast_mask = %mask : !llvm.ptr<7>
  // LLVM: call void @llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // PTX: mbarrier.arrive.expect_tx.shared::cluster.multicast::cluster::32b.b64 _, [{{%rd[0-9]+}}], {{%r[0-9]+}}, {{%r[0-9]+}};
  nvvm.mbarrier.arrive.expect_tx %bar, %count multicast_mask = %mask : !llvm.ptr<7>, i32, i32
  // LLVM: call void @llvm.nvvm.mbarrier.arrive.drop.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 %2, i1 true)
  // PTX: mbarrier.arrive_drop.expect_tx.shared::cluster.multicast::cluster::32b.b64 _, [{{%rd[0-9]+}}], {{%r[0-9]+}}, {{%r[0-9]+}};
  nvvm.mbarrier.arrive_drop.expect_tx %bar, %count multicast_mask = %mask : !llvm.ptr<7>, i32, i32
  llvm.return
}

// LLVM-LABEL: @constant_masks
// PTX-LABEL: constant_masks(
llvm.func @constant_masks(%bar: !llvm.ptr<7>, %count: i32) {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %high = llvm.mlir.constant(-2147483648 : i32) : i32
  // LLVM: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 1, i32 0, i1 true)
  // PTX: mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64
  nvvm.mbarrier.arrive %bar multicast_mask = %zero : !llvm.ptr<7>
  // LLVM: call void @llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1, i32 -2147483648, i1 true)
  // PTX: mbarrier.expect_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64
  nvvm.mbarrier.expect_tx %bar, %count multicast_mask = %high : !llvm.ptr<7>, i32, i32
  // LLVM: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 1, i32 0, i1 false)
  // PTX: mbarrier.arrive.shared::cluster.b64
  nvvm.mbarrier.arrive %bar : !llvm.ptr<7>
  llvm.return
}
