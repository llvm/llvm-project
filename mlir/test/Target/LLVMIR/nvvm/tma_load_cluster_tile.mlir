// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_load_1d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_1d_all_tile(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i16 %4, i64 %5) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 0, i64 %5, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 %4, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 %4, i64 %5, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 0, i64 %5, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 %4, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 %4, i64 %5, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 0, i64 %5, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 %4, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i16 %4, i64 %5, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_2d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_2d_all_tile(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i16 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 0, i64 %6, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 %5, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 %5, i64 %6, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 0, i64 %6, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 %5, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 %5, i64 %6, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 0, i64 %6, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 %5, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i16 %5, i64 %6, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_3d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_3d_all_tile(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i16 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 0, i64 %7, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i64 %7, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 0, i64 %7, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i64 %7, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 0, i64 %7, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i64 %7, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_4d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_4d_all_tile(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 %8) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 0, i64 %8, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 %8, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 0, i64 %8, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 %8, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 0, i64 %8, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i64 %8, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_5d_all(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %ctamask: i16, %cache: i64) {
  // CHECK-LABEL: define void @tma_load_5d_all(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 %9, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 %9, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 %9, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] multicast_mask = %ctamask : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] l2_cache_hint = %cache : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] multicast_mask = %ctamask l2_cache_hint = %cache : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] l2_cache_hint = %cache {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] multicast_mask = %ctamask l2_cache_hint = %cache {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] l2_cache_hint = %cache {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] multicast_mask = %ctamask l2_cache_hint = %cache {mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_2d_tile_gather4(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %row0: i32, %col0: i32, %col1: i32, %col2: i32, %col3: i32, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_2d_tile_gather4(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 %9, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 %9, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 0, i64 %9, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.gather4.2d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i64 %9, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<tile_gather4>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}
