// RUN: mlir-opt -split-input-file -verify-diagnostics %s
// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_cp_async_bulk_global_to_shared_cluster
llvm.func @llvm_nvvm_cp_async_bulk_global_to_shared_cluster(%dst : !llvm.ptr<3>, %src : !llvm.ptr<1>, %mbar : !llvm.ptr<3>, %size : i32, %mc : i16, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(3) %[[DST:.*]], ptr addrspace(3) %[[MBAR:.*]], ptr addrspace(1) %[[SRC:.*]], i32 %[[SIZE:.*]], i16 0, i64 0, i1 false, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(3) %[[DST]], ptr addrspace(3) %[[MBAR]], ptr addrspace(1) %[[SRC]], i32 %[[SIZE]], i16 0, i64 %[[CH:.*]], i1 false, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(3) %[[DST]], ptr addrspace(3) %[[MBAR]], ptr addrspace(1) %[[SRC]], i32 %[[SIZE]], i16 %[[MC:.*]], i64 0, i1 true, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(3) %[[DST]], ptr addrspace(3) %[[MBAR]], ptr addrspace(1) %[[SRC]], i32 %[[SIZE]], i16 %[[MC]], i64 %[[CH]], i1 true, i1 true)
  nvvm.cp.async.bulk.shared.cluster.global %dst, %src, %mbar, %size : !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.shared.cluster.global %dst, %src, %mbar, %size l2_cache_hint = %ch : !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.shared.cluster.global %dst, %src, %mbar, %size multicast_mask = %mc : !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.shared.cluster.global %dst, %src, %mbar, %size multicast_mask = %mc l2_cache_hint = %ch : !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cp_async_bulk_shared_cta_to_shared_cluster
llvm.func @llvm_nvvm_cp_async_bulk_shared_cta_to_shared_cluster(%dst : !llvm.ptr<3>, %src : !llvm.ptr<3>, %mbar : !llvm.ptr<3>, %size : i32) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.shared.cta.to.cluster(ptr addrspace(3) %0, ptr addrspace(3) %2, ptr addrspace(3) %1, i32 %3)
  nvvm.cp.async.bulk.shared.cluster.shared.cta %dst, %src, %mbar, %size : !llvm.ptr<3>, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cp_async_bulk_shared_cta_to_global
llvm.func @llvm_nvvm_cp_async_bulk_shared_cta_to_global(%dst : !llvm.ptr<1>, %src : !llvm.ptr<3>, %size : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.shared.cta.to.global(ptr addrspace(1) %[[DST:.*]], ptr addrspace(3) %[[SRC:.*]], i32 %[[SIZE:.*]], i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.shared.cta.to.global(ptr addrspace(1) %[[DST:.*]], ptr addrspace(3) %[[SRC:.*]], i32 %[[SIZE:.*]], i64 %[[CH:.*]], i1 true)
  nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size : !llvm.ptr<1>, !llvm.ptr<3>

  nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch : !llvm.ptr<1>, !llvm.ptr<3>
  llvm.return
}
