// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_load_3d_im2col(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %off0: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_3d_im2col(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 %8) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 0, i64 %8, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 %8, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 0, i64 %8, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 %8, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 0, i64 %8, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 %8, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_4d_im2col(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %mask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_4d_im2col(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] multicast_mask = %mask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] multicast_mask = %mask {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] multicast_mask = %mask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] multicast_mask = %mask {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] multicast_mask = %mask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] multicast_mask = %mask {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_5d_im2col(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %off0: i16, %off1: i16, %off2: i16, %mask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_5d_im2col(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 %12) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 %12, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 0, i64 %12, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 %12, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 0, i64 %12, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 %12, i1 true, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 0, i64 %12, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] multicast_mask = %mask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] multicast_mask = %mask {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr
  
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] multicast_mask = %mask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] multicast_mask = %mask {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] multicast_mask = %mask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
 
 nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] multicast_mask = %mask {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%off0, %off1, %off2] {mode = #nvvm.tma_load_mode<im2col>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_3d_im2col_w(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %wHalo: i16, %wOffset: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_3d_im2col_w(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 %9, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 %9, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 %9, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_4d_im2col_w(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %wHalo: i16, %wOffset: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_4d_im2col_w(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_5d_im2col_w(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %wHalo: i16, %wOffset: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_5d_im2col_w(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 %11, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 %11, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 %11, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_3d_im2col_w_128(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %wHalo: i16, %wOffset: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_3d_im2col_w_128(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 %9, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 %9, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 0, i64 %9, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.3d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_4d_im2col_w_128(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %wHalo: i16, %wOffset: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_4d_im2col_w_128(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 0, i64 %10, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.4d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i16 %7, i16 %8, i16 %9, i64 %10, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}

llvm.func @tma_load_5d_im2col_w_128(%tma: !llvm.ptr, %dest: !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %wHalo: i16, %wOffset: i16, %ctamask: i16, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_5d_im2col_w_128(ptr %0, ptr addrspace(7) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 0, i1 false, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 0, i1 true, i1 false, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 %11, i1 false, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11, i1 true, i1 true, i32 0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 0, i1 false, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 0, i1 true, i1 false, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 %11, i1 false, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11, i1 true, i1 true, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 0, i1 false, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 0, i1 true, i1 false, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 0, i64 %11, i1 false, i1 true, i32 2)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.w.128.5d(ptr addrspace(7) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i64 %11, i1 true, i1 true, i32 2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] im2col[%wHalo, %wOffset] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col_w_128>, group = #nvvm.cta_group<cta_2>} : !llvm.ptr<7>, !llvm.ptr
  llvm.return
}
